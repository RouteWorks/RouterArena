from sentence_transformers import SentenceTransformer


class HybridRouterEncoder:
    def __init__(self):
        """
        Loads "all-mpnet-base-v2" once on construction and stores it as self.model.
        The model is set to eval mode and has gradients disabled.
        """
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        self.model.eval()
        # Gradient disabling is handled by sentence_transformers internally, but we make it explicit here.
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, text: str | list[str]):
        """
        Accepts either a single string or a list of strings, and always returns a numpy array.
        For a single string, the shape should be (768,) for a list of N strings, it should be (N, 768)
        """
        return self.model.encode(text, show_progress_bar=False)
