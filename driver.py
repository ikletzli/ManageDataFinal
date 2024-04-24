import pandas as pd
import numpy as np
import re

from tqdm import tqdm
import argparse

# Data Loading Methods =================================================================================================

def LoadData(add_sample, auth_sample):
    '''
    Wrapper method to load both data sets
    '''
    author_data = LoadAuthors()
    address_data = LoadAddresses()
    
    if add_sample != None and int(add_sample) < len(address_data):
        address_data = address_data.sample(n=int(add_sample), random_state=42)
        
    if auth_sample != None and int(auth_sample) < len(author_data):
        author_data = author_data.sample(n=int(auth_sample), random_state=42)

    # uncomment to test very small example
    #author_data = {'isbn': [1, 1, 1, 2, 2, 2], 'author': ["Mary Lee", "Lee, Mary", "M. Lee", "Smith, James", "James Smith", "J. Smith"]}
    #author_data = pd.DataFrame(data=author_data)

    address_data = address_data.groupby('ein')['address'].agg(set= lambda x: set(x))
    author_data = author_data.groupby('isbn')['author'].agg(set= lambda x: set(x))

    return author_data, address_data


def LoadAddresses():
    '''
    Helper method to load the addresses data set
    '''
    address_df = pd.read_csv('data/address.csv')
    address_df = address_df[['EIN', 'Street Address 1']]

    address_df = address_df.drop(address_df[address_df['Street Address 1'].isnull()].index)
    address_df = address_df.drop(address_df[address_df['EIN'].isnull()].index)

    address_df['EIN'] = address_df['EIN'].astype('string')
    address_df["EIN"] = address_df["EIN"].map(lambda x: re.sub(r'\D', '', x))
    address_df['EIN'] = address_df['EIN'].astype('int64')

    address_df = address_df.rename(columns={'EIN': 'ein', 'Street Address 1': 'address'})
    address_df = address_df.fillna('')
    
    return address_df


def LoadAuthors():
    '''
    Helper method to load the authors data set
    '''
    book_df = pd.read_csv('data/book.txt', sep="\t", header=None, names=['publisher', 'isbn', 'title', 'author'])
    book_df = book_df.drop('publisher', axis=1)
    book_df = book_df.drop('title', axis=1)

    book_df = book_df.fillna('')

    return book_df


# Unsupervised Grouping Methods ========================================================================================

def GeneratingCandidateReplacements(data):
    '''
    Creates the sets of replacement candidates from the relevent columns of the data set

    arguments: authors data set, addresses data set
    return value: list of candidate pairs for both data sets (author candidates, address candidates)
    '''
    # Paper says maybe we shouldn't include identical values in the candidate replacements,
    #   so maybe add
    #     if author_list[i] != author_list[j]
    #   above line 56
    #   and the respective line for adresses above line 64

    candidates = []
    for index in tqdm(range(len(data)), 'generating author candidates'):
        candidate_list = list(data.iloc[index, 0])
        for i in range(len(candidate_list) - 1):
            for j in range(i+1, len(candidate_list)):
                candidates.append((candidate_list[i], candidate_list[j]))
            
    return candidates


# Graph Construction Methods ===========================================================================================

# Algorithm 1
def GoldenRecordCreation(data):
    candidates = GeneratingCandidateReplacements(data)
    print(candidates)
    grouping = UnsupervisedGrouping(candidates)
    sorted_grouping = dict(sorted(grouping.items(), key=lambda x: len(x[1]), reverse=True))
    
# Algorithm 2
def UnsupervisedGrouping(candidates):
    graphs = BuildTransformationGraph(candidates)
    author_inverted = InvertedIndexAlg2(graphs)

    grouping = {}
    for graph in graphs:
        n_last = 0
        for key in graph.keys():
            if key[1] > n_last:
                n_last = key[1]
                
        pmax, lmax = SearchPivot(graph, "", graphs, 0, None, [], n_last, author_inverted)
        
        if pmax in grouping:
            grouping[pmax] += [graph]
        else:
            grouping[pmax] = [graph]

    return grouping

def SearchPivot(graph, path, graphs, n_first, pmax, lmax, n_last, inverted):
    
    if n_first == n_last:
        if len(graphs) > len(lmax):
            return path, graphs
    else:
        for edge, str_functions in graph.items():
            for str_fun in str_functions:
                p_prime = path + str_fun
                l_prime = []
                if n_first == 0:
                    l_prime = inverted[str_fun] # [(i, edge[0], edge[1])]
                else:
                    for g1, i1, j1 in graphs:
                        for g2, i2, j2 in inverted[str_fun]:
                            if g1 == g2:
                                if j1 == i2:
                                    l_prime.append(g1, i1, j2)
                                    
                pmax, lmax = SearchPivot(graph, p_prime, l_prime, edge[1], pmax, lmax, n_last, inverted)

    return pmax, lmax

def BuildTransformationGraph(candidates):
    '''
    Algorithm 8
    Turns each set of candidates into a graph representing the transformation graph between those candidates

    arguments: list of candidates in the form (candidate 1, candidate 2) where both values are strings
    return value: set of graphs G
    '''
    graphs = []
    for candidate in tqdm(candidates, 'generating transformation graphs'):
        pre_defined_regex = [r"[A-Z]+", r"[a-z]+", r"\s+", r"[0-9]+"]
        matches_0 = {r"[A-Z]+": [], r"[a-z]+": [], r"\s+": [], r"[0-9]+": []}
        matches_1 = {r"[A-Z]+": [], r"[a-z]+": [], r"\s+": [], r"[0-9]+": []}
        # now only finds full matches i.e "abc" and doesn't separate into "abc" "ab" "bc" "a" "b" "c"
        # might need to make that change but not sure
        for regex in pre_defined_regex:
            for match in re.finditer(regex, candidate[0]):
                matches_0[regex] += [match]
            for match in re.finditer(regex, candidate[1]):
                matches_1[regex] += [match]

        P_0 = {}
        P_1 = {}
        for match_dict, P in [(matches_0, P_0), (matches_1, P_1)]:
            for regex, match_list in match_dict.items():
                for i in range(len(match_list)):
                    match = match_list[i]

                    x_str_one = "match" + regex + str(i+1) + "B"
                    x_str_two = "match" + regex + str(i-len(match_list)) + "B"

                    y_str_one = "match" + regex + str(i+1) + "E"
                    y_str_two = "match" + regex + str(i-len(match_list)) + "E"

                    if match.start() not in P:
                        P[match.start()] = [x_str_one, x_str_two]
                    else:
                        P[match.start()] += [x_str_one, x_str_two]

                    if match.end() not in P:
                        P[match.end()] = [y_str_one, y_str_two]
                    else:
                        P[match.end()] += [y_str_one, y_str_two]

        for index, P in [(0, P_0), (1, P_1)]:
            for i in range(len(candidate[index]) + 1):
                k = i+1
                str_one = "constpos" + str(k)
                str_two = "constpos" + str(k - len(candidate[index]) - 2)

                if i not in P:
                    P[i] = [str_one, str_two]
                else:
                    P[i] += [str_one, str_two]

        graph_0 = {}
        graph_1 = {}

        for i in range(len(candidate[0])):
            for j in range(i+1, len(candidate[0]) + 1):
                sub = candidate[0][i:j]
                graph_0[(i,j)] = ["conststr" + sub]

        for i in range(len(candidate[1])):
            for j in range(i+1, len(candidate[1]) + 1):
                sub = candidate[1][i:j]
                graph_1[(i,j)] = ["conststr" + sub]
                for match in re.finditer(re.escape(sub), candidate[0]):
                    for f in P_0[match.start()]:
                        for g in P_0[match.end()]:
                            graph_1[(i,j)] += ["sub" + f + g]
                    for f in P_1[i]:
                        for g in P_1[j]:
                            graph_0[(match.start(),match.end())] += ["sub" + f + g]

        graphs += [graph_0, graph_1]
    
    return graphs

def InvertedIndexAlg2(graphs):
    '''
    Helper method to create an inverted index structure over the provided graphs
    Used in Algorithms 2, 6

    arguments: set of graphs
    return value: dictionary containing index mapping
    '''
    I = {}

    for i in range(len(graphs)):
        graph = graphs[i]
        for edge in graph:
            edge_labels = graph[edge]
            for edge_label in edge_labels:
                if edge_label in I:
                    I[edge_label] += [(i, edge[0], edge[1])]
                else:
                    I[edge_label] = [(i, edge[0], edge[1])]

    return I

def InvertedIndex(graphs):
    '''
    Helper method to create an inverted index structure over the provided graphs
    Used in Algorithms 2, 6

    arguments: set of graphs
    return value: dictionary containing index mapping
    '''
    I = {}

    for graph in graphs:
        for edge in graph:
            edge_labels = graph[edge]
            for edge_label in edge_labels:
                if edge_label in I:
                    I[edge_label].append(edge_labels)
                else:
                    I[edge_label] = [edge_labels]

    return I


# Incremental Grouping Methods =========================================================================================

def IncrementalGrouping():
    '''
    Algorithm 5
    Place holder method containing the function calls required to implement incremental grouping in the main algorithm
    '''
    # replace line 4 of Algorithm 1 (GoldenRecordCreation)
    G = Preprocessing([])  # replace arg with set of candidates

    # replace line 6 of Algorithm 1
    sigma = GenerateNextLargestGroup(G)

    # replace line 9 of Algorithm 1
    G = UpdateGraph(G, sigma)


def Preprocessing(phi):
    '''
    Algorithm 6
    Generates the bounds for the set of graphs G corresponding to the set of candidate replacements phi
    Differs from BuildTransformationGraph in that it allows the use of Algorithm 7

    arguments: set of replacement candidates phi
    return value: set of graphs, and a set of upper bounds for those graphs
    '''
    graphs = BuildTransformationGraph(phi)
    I = InvertedIndex(graphs)
    bounds = {}

    for graph in graphs:
        # initialize upper-bound list
        largest_index = 0
        for edge in graph:
            if edge[1] > largest_index:
                largest_index = edge[1]
        upper_bounds = []
        for i in range(largest_index):
            upper_bounds.append(0)

        for edge in graph:
            for label in edge:
                for k in range(edge[0], edge[1]):
                    if upper_bounds[k] < len(I[label]):
                        upper_bounds[k] = len(I[label])

        bounds[graph] = min(upper_bounds)
    
    return graphs, bounds


def GenerateNextLargestGroup(G, upper_bounds):
    '''
    Algorithm 7
    Analyzes the current set of graphs and determines the next largest group of similar transformations

    arguments: set of graphs G, the corresponding set of upper bounds
    return value: list of graphs containing the current longest path in G
    '''
    l = []
    return l


def UpdateGraph(G, sigma):
    '''
    Helper method used to update the set of graphs G
    Corresponds to the psuedo code description of line 3 of Algorithm 5

    arguments: set of graphs G, group of similar replacements sigma
    return value: an updated G
    '''
    return G


# Driver ===============================================================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="My Python script with options -a and -b")
    parser.add_argument("--address", help="Enable option -a")
    parser.add_argument("--author", help="Enable option -b")

    args = parser.parse_args()

    add_sample = None
    auth_sample = None
    if args.address:
        add_sample = args.address
    if args.author:
        auth_sample = args.author
    
    author_data, address_data = LoadData(add_sample, auth_sample)
    
    GoldenRecordCreation(author_data)