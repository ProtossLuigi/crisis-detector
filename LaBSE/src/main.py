import nlp_ws

from src.worker import GoemoLabseWorker

if __name__ == "__main__":
    nlp_ws.NLPService.main(GoemoLabseWorker)
