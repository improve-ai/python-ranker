from io import BytesIO
import gzip
import os
from typing import Union


def is_gz_bytes(chkd_bytes: bytes) -> bool:
    """
    Checks if provided bytes is a gz zipped file stream. If so returns True

    Parameters
    ----------
    chkd_bytes: bytes
        bytes to check for being gz zipped file

    Returns
    -------
    bool
        flag indicating if provided bytes is a gz compressed bytes

    """

    if not isinstance(chkd_bytes, bytes):
        return False

    with gzip.GzipFile(fileobj=BytesIO(chkd_bytes)) as f:
        try:
            f.read(1)
            return True
        except Exception as exc:
            return False


def get_unzip_gz(to_be_unzpd_bytes: bytes) -> bytes:
    """
    Unzips provided input bytes

    Parameters
    ----------
    to_be_unzpd_bytes: bytes
        bytes to be unzipped

    Returns
    -------
    bytes
        unzipped bytes

    """

    if not isinstance(to_be_unzpd_bytes, bytes):
        raise TypeError('`unzpd_bytes` param should be of bytes type')

    unzpd_buffer = BytesIO(to_be_unzpd_bytes)
    unzpd_buffer.seek(0)
    res = gzip.decompress(unzpd_buffer.read())
    return res


def check_and_get_unzipped_model(model_src: Union[str, bytes]) -> Union[str, bytes]:
    """
    Checks if provided model is a compressed one and unzips it if so.

    Parameters
    ----------
    model_src: Union[str, bytes]
        model source; cna be bytes or string

    Returns
    -------
    Union[str, bytes]
        either decompressed bytes or path to model file

    """

    if isinstance(model_src, bytes):
        if is_gz_bytes(chkd_bytes=model_src):
            return get_unzip_gz(to_be_unzpd_bytes=model_src)
        else:
            return model_src
    elif isinstance(model_src, str):
        if not os.path.isfile(model_src):
            raise ValueError(
                'This is not a proper path: {} and reading model from '
                'string is not supported'.format(model_src))

        try:
            with open(model_src, 'rb') as chkd_bytes:
                read_chkd_bytes = chkd_bytes.read()
                if is_gz_bytes(chkd_bytes=read_chkd_bytes):
                    print('Returning unzipped file')
                    return gzip.decompress(read_chkd_bytes)
        except Exception as exc:
            print(
                'When checking file: {} the following error occured'
                .format(model_src, exc))
        return model_src
    else:
        raise TypeError(
            'Unsupported model source type: {}'.format(type(model_src)))
