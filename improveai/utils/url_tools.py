import requests as rq
from urllib.parse import urlparse


def is_path_http_addr(path: str) -> bool:
    """
    Checks if provided string is a http address.

    Parameters
    ----------
    path: str or Path
        is this path a URL?

    Returns
    -------
    bool
        True if path is a URL otherwise False

    """
    try:
        res = urlparse(path)
        return all([res.scheme, res.netloc])
    except ValueError:
        return False


def get_model_bytes_from_url(model_url: str) -> bytes:
    """
    Gets model from provided URL.

    Parameters
    ----------
    model_url: str
        url to get the model from

    Returns
    -------
    bytes
        downloaded bytes source url

    """

    try:
        headers = {"Accept-Encoding": "identity"}
        model_resp = rq.get(model_url, headers=headers)
        if model_resp.status_code != 200:
            raise ('Unable to load model from: {} path'.format(model_url))
    except Exception as exc:
        print('While getting model got: {} error'.format(exc))
        raise ValueError('Unable to load model from: {} path'.format(model_url))

    dled_model = model_resp.content

    if not isinstance(model_resp.content, bytes):
        raise TypeError('Downloaded model is not of a `bytes` type')

    return dled_model
