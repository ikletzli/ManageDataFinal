import pandas as pd
import numpy as np
import re

# # Data Loading Methods =================================================================================================

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

if __name__ == "__main__":
    author_data, address_data = LoadData()

    author_candidates, address_candidates = GeneratingCandidateReplacements(author_data, address_data)

    print(author_data)
    print(address_data)