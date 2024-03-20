import pandas as pd


# Data Loading Methods =================================================================================================

def LoadData():
    author_data = LoadAuthors()
    address_data = LoadAddresses()

    return author_data, address_data


def LoadAuthors():
    return set()


def LoadAddresses():
    return set()
