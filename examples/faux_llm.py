"""
faux_llm.py - Un faux LLM pour la CI
====================================

Répond comme l'API d'Ollama, sans modèle : la CI vérifie ainsi tout le parcours de
genai_example.py en quelques secondes. Les notes des juges n'ont donc aucune valeur ici :
ce serveur contrôle la mécanique, pas la qualité d'un assistant.

Exécution:
---------
docker compose exec -d runner python faux_llm.py
docker compose exec -e OLLAMA_BASE_URL=http://localhost:11434/v1 runner python genai_example.py
"""

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def repondre(messages):
    """Note 'yes' à la place d'un juge ; sinon renvoie la fiche, qui contient la réponse."""
    texte = " ".join(str(m["content"]) for m in messages)
    if "rationale" in texte:
        return json.dumps({"rationale": "Faux LLM : aucune analyse.", "result": "yes"})
    return re.sub(r".*d'après cette fiche : ", "", messages[0]["content"], flags=re.S)


class Serveur(BaseHTTPRequestHandler):
    def do_POST(self):
        corps = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        contenu = json.dumps({
            "id": "faux",
            "object": "chat.completion",
            "created": 0,
            "model": corps["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": repondre(corps["messages"])},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(contenu)))
        self.end_headers()
        self.wfile.write(contenu)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 11434), Serveur).serve_forever()
