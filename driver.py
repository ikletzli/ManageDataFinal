import pandas as pd
import numpy as np
import re


# Data Loading Methods =================================================================================================

def LoadData():
    author_data = LoadAuthors()
    address_data = LoadAddresses()

    return author_data, address_data


def LoadAddresses():
    address_df = pd.read_csv('data/address.csv')
    address_df = address_df[['EIN', 'Street Address 1']]

    address_df = address_df.drop(address_df[address_df['Street Address 1'].isnull()].index)
    address_df = address_df.drop(address_df[address_df['EIN'].isnull()].index)

    address_df['EIN'] = address_df['EIN'].astype('string')
    address_df["EIN"] = address_df["EIN"].map(lambda x: re.sub(r'\D', '', x))
    address_df['EIN'] = address_df['EIN'].astype('int64')

    address_df = address_df.rename(columns={'EIN': 'ein', 'Street Address 1': 'address'})

    address_df = address_df.groupby('ein')['address'].agg(set= lambda x: set(x))

    return address_df


def LoadAuthors():
    book_df = pd.read_csv('data/book.txt', sep="\t", header=None, names=['publisher', 'isbn', 'title', 'author'])
    book_df = book_df.drop('publisher', axis=1)
    book_df = book_df.drop('title', axis=1)

    book_df = book_df.groupby('isbn')['author'].agg(set= lambda x: set(x))

    return book_df


def GeneratingCandidateReplacements(author_data, address_data):

    # Paper says maybe we shouldn't include identical values in the candidate replacements,
    #   so maybe add
    #     if author_list[i] != author_list[j]
    #   above line 56
    #   and the respective line for adresses above line 64

    author_candidates = []
    for index in range(len(author_data)):
        author_list = list(author_data.iloc[index, 0])
        for i in range(len(author_list) - 1):
            for j in range(i+1, len(author_list)):
                author_candidates.append((author_list[i], author_list[j]))
                author_candidates.append((author_list[j], author_list[i]))

    address_candidates = []
    for index in range(len(address_data)):
        address_list = list(address_data.iloc[index, 0])
        for i in range(len(address_list) - 1):
            for j in range(i+1, len(address_list)):
                address_candidates.append((address_list[i], address_list[j]))
                address_candidates.append((address_list[j], address_list[i]))
            
    return author_candidates, address_candidates


# Incremental Grouping Methods =========================================================================================

def IncrementalGrouping():
    '''
    Algorithm 5
    Place holder method containing the function calls required to implement incremental grouping in the main algorithm
    '''
    # replace line 4 of Algorithm 1 (GoldenRecordCreation)
    G = Preprocessing()

    # replace line 6 of Algorithm 1
    sigma = GenerateNextLargestGroup(G)

    # replace line 9 of Algorithm 1
    G = UpdateGraph(G, sigma)

    return


def Preprocessing(phi):
    '''
    Algorithm 6
    Generates the set of graphs G corresponding to the set of candidate replacements phi

    arguments:
    return value:
    '''
    return


def GenerateNextLargestGroup(G):
    '''
    Algorithm 7
    Desc

    arguments:
    return value:
    '''
    return


def UpdateGraph(G, sigma):
    '''
    Desc

    arguments:
    return value:
    '''
    return


# Driver ===============================================================================================================

if __name__ == "__main__":
    author_data, address_data = LoadData()

    author_candidates, address_candidates = GeneratingCandidateReplacements(author_data, address_data)

    print(author_data)
    print(address_data)