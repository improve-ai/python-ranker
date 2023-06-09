from io import BytesIO
import gzip
import os
from pathlib import Path
from typing import Union


def is_gz_bytes(checked_bytes: bytes) -> bool:
    """
    Checks if provided bytes is a gzipped file stream. If so returns True.

    Parameters
    ----------
    checked_bytes: bytes
        bytes to check for being gz zipped file

    Returns
    -------
    bool
        flag indicating if provided bytes is a gz compressed bytes
    """

    if not isinstance(checked_bytes, bytes):
        return False

    with gzip.GzipFile(fileobj=BytesIO(checked_bytes)) as f:
        try:
            f.read(1)
            return True
        except Exception as exc:
            return False


def get_unzipped_gz(to_be_unzipped_bytes: bytes) -> bytes:
    """
    Unzips provided input bytes.

    Parameters
    ----------
    to_be_unzipped_bytes: bytes
        bytes to be unzipped

    Returns
    -------
    bytes
        unzipped bytes
    """

    if not isinstance(to_be_unzipped_bytes, bytes):
        raise TypeError('`unzpd_bytes` param should be of bytes type')

    unzipped_buffer = BytesIO(to_be_unzipped_bytes)
    unzipped_buffer.seek(0)
    res = gzip.decompress(unzipped_buffer.read())
    return res


def check_and_get_unzipped_model(model_src: Union[str, bytes, Path]) -> Union[str, bytes, Path]:
    """
    Checks if provided model is a gzipped one and unzips it if so.

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
        if is_gz_bytes(checked_bytes=model_src):
            return get_unzipped_gz(to_be_unzipped_bytes=model_src)
        else:
            return model_src
    elif isinstance(model_src, str) or isinstance(model_src, Path):
        if str(model_src).startswith('~'):
            # append the absolute prefix to the input path
            abs_model_src = os.path.expanduser(model_src)
            if isinstance(model_src, Path):
                # if model_src was of a Path type then we need to convert it back
                abs_model_src = Path(abs_model_src)
            # update model_src value
            model_src = abs_model_src
            print(f'Model src: {model_src}')
        if not os.path.isfile(model_src):
            raise FileNotFoundError(
                'This is not a proper path: {} and reading model from '
                'string is not supported'.format(model_src))

        try:
            with open(model_src, 'rb') as chkd_bytes:
                read_chkd_bytes = chkd_bytes.read()
                if is_gz_bytes(checked_bytes=read_chkd_bytes):
                    return gzip.decompress(read_chkd_bytes)
        except Exception as exc:
            print(
                'When checking file: {} the following error occurred'
                .format(model_src, exc))
        return model_src
    else:
        raise TypeError(
            'Unsupported model source type: {}'.format(type(model_src)))
