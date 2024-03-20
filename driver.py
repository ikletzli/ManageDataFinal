import pandas as pd

def LoadData():
    author_data = LoadAuthors()
    address_data = LoadAddresses()

    return author_data, address_data


def LoadAuthors():
    return {}


def LoadAddresses():
    return {}