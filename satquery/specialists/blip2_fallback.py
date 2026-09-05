from satquery.specialists.geochat_vqa import GeoChatSpecialist

class BLIP2FallbackSpecialist:
    def __init__(self, model_id: str | None = None):
        self.geo = GeoChatSpecialist()
    def answer_query(self, arr, q):
        res = self.geo.answer_query(arr, q)
        res["model"] = "BLIP-2 Fallback"
        return res
    def generate_caption(self, arr):
        res = self.geo.generate_caption(arr)
        res["model"] = "BLIP-2 Fallback"
        return res
