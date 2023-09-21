class S2Bands:
    def __init__(self):
        pass

    @staticmethod
    def get_R10m_bands():
        return ["B08"]

    @staticmethod
    def get_R20m_bands():
        return ["B01","B02","B03","B04","B05","B06","B07","B8A","B11","B12"]

    @staticmethod
    def get_R60m_bands():
        return ["B09"]

    @staticmethod
    def get_all_bands():
        return S2Bands.get_R20m_bands() + S2Bands.get_R10m_bands() + S2Bands.get_R60m_bands()
