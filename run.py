"""Entrypoint: python run.py"""
import logging
import uvicorn

from kaglaw.config import HOST, PORT


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run("kaglaw.web.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
