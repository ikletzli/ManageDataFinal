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
    # author_data = {'isbn': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'author': ["Mary Lee", "Lee, Mary", "M. Lee", "Smith, James", "James Smith", "J. Smith", "Jim Choo", "Choo, Jim", "J. Choo"]}
    # author_data = pd.DataFrame(data=author_data)

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
    candidates = []
    for index in tqdm(range(len(data)), 'generating author candidates'):
        candidate_list = list(data.iloc[index, 0])
        for i in range(len(candidate_list) - 1):
            for j in range(i+1, len(candidate_list)):
                candidates.append((candidate_list[i], candidate_list[j]))
            
    return candidates


# Graph Construction Methods ===========================================================================================

def GetSubstringIndex(sub, val):
    return_val = None
    # points to const position
    if sub.startswith("pos"):
        const = int(sub[3:])
        if const < 0:
            return_val = len(val) + const + 1
        else:
            return_val = const - 1
    # points to regex match
    else:
        parsed = sub[:-1]
        parsed = parsed.split("+")
        parsed[0] += "+"
        parsed += sub[-1]
        
        regex = parsed[0]
        match_num = int(parsed[1])
        
        # either "B" or "E" for Beginning or End
        match_pos = parsed[2]
        
        # find all matches of regex
        matches = []
        for match in re.finditer(regex, val):
            matches += [match]
        
        # find 1st last etc. match of regex based on match_num    
        index = 0
        if match_num > 0:
            index = match_num - 1
        else:
            index = len(matches) + match_num

        # get index to start or end of match based on match_pos
        if match_pos == "B":
            return_val = matches[index].start()
        else:
            return_val = matches[index].end()

    return return_val

def ApplyGrouping(values, transform_str):
    for val in values:
        if transform_str != None:
            # split by transformation type
            steps = transform_str.replace("conststr", "sub").split("sub")
            new_str = ""
            for step in steps:
                if step != "":
                    # sub string match based on regex or const position
                    if step.startswith("match") or step.startswith("constpos"):
                        sub = step.replace("match", "const").split("const")
                        sub = list(filter(None, sub))
                        
                        # sub[0] contains first position and sub[0] contains second position for substring
                        first = GetSubstringIndex(sub[0], val[0])
                        second = GetSubstringIndex(sub[1], val[0])

                        new_str += val[0][first:second]
                    # const string transformation
                    else:
                        new_str += step
                    
            print("\t", val[0], "|", val[1], "|", new_str)


# Algorithm 1
def GoldenRecordCreation(data):
    candidates = GeneratingCandidateReplacements(data)
    grouping = UnsupervisedGrouping(candidates)
    sorted_grouping = dict(sorted(grouping.items(), key=lambda x: len(x[1]), reverse=True))
    for key, value in sorted_grouping.items():
        print(key)
        ApplyGrouping(value, key)
    
# Algorithm 2
def UnsupervisedGrouping(candidates):
    graphs = BuildTransformationGraph(candidates,True)
    inverted = InvertedIndexAlg2([graph[2] for graph in graphs])
    
    # line 2 from Algorithm 4
    global glo
    glo = [1] * len(graphs)
    grouping = {}
    
    index = 0
    for str, replace, graph in tqdm(graphs,"finding pivot paths"):
        n_last = 0
        for key in graph.keys():
            if key[1] > n_last:
                n_last = key[1]
        
        pmax, lmax = SearchPivot(graph, "", graphs, 0, None, [], n_last, inverted, index)
        index += 1
        if pmax in grouping:
            grouping[pmax] += [(str, replace)]
        else:
            grouping[pmax] = [(str, replace)]

    return grouping

def SearchPivot(graph, path, graphs, n_first, pmax, lmax, n_last, inverted, g_i):
    global glo
    
    # transformed graph completely
    if n_first == n_last:
        new_graphs = []
        
        # only keep other graphs that were complete transformations
        for g, i, j, l in graphs:
            if j == l:
                new_graphs.append((g,i,j,l))
        
        # lines 3 and 4 from Algorithm 4
        for g, _, _, _ in new_graphs:
            if glo[g] < len(new_graphs):
                glo[g] = len(new_graphs)
                
        if len(new_graphs) > len(lmax):
            return path, new_graphs
    else:
        for edge, str_functions in graph.items():
            if edge[0] < n_first:
                continue
            
            for str_fun in str_functions:
                p_prime = path + str_fun
                l_prime = []
                
                # valid graphs must have the str_fun and start at 0
                if n_first == 0:
                    for g, i, j, l in inverted[str_fun]:
                        if i == 0:
                            l_prime.append((g,i,j,l)) # [(i, edge[0], edge[1], last)]
                else:
                    for g1, i1, j1, l1 in graphs:
                        for g2, i2, j2, l2 in inverted[str_fun]:
                            if g1 == g2:
                                if j1 == i2:
                                    l_prime.append((g1, i1, j2, l2))
                
                if len(l_prime) > len(lmax) and len(l_prime) >= glo[g_i]: # line 5 Algorithm 4
                    pmax, lmax = SearchPivot(graph, p_prime, l_prime, edge[1], pmax, lmax, n_last, inverted, False)

    return pmax, lmax

def BuildTransformationGraph(candidates, keep_strings=False):
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
        
        # find all matches of the 4 predefined regexes
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

                    # lines 5 and 6
                    x_str_one = "match" + regex + str(i+1) + "B"
                    x_str_two = "match" + regex + str(i-len(match_list)) + "B"
                    
                    # lines 7 and 8
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
                
                # lines 10 and 11
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
                    # add all substring transformations to either graph
                    for f in P_0[match.start()]:
                        for g in P_0[match.end()]:
                            graph_1[(i,j)] += ["sub" + f + g]
                    for f in P_1[i]:
                        for g in P_1[j]:
                            graph_0[(match.start(),match.end())] += ["sub" + f + g]

        if keep_strings:
            # tells how to transform graphs[0] to graphs[1] using graphs[2]
            graphs += [(candidate[1], candidate[0], graph_0), (candidate[0], candidate[1], graph_1)]
        else:    
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
        
        # get the last index in this graph
        n_last = 0
        for key in graph.keys():
            if key[1] > n_last:
                n_last = key[1]
                
        for edge in graph:
            edge_labels = graph[edge]
            for edge_label in edge_labels:
                if edge_label in I:
                    # keep track of graph, edge, and len of the graph containing this edge_label
                    I[edge_label] += [(i, edge[0], edge[1], n_last)]
                else:
                    I[edge_label] = [(i, edge[0], edge[1], n_last)]

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
    
    # setup command line args
    parser = argparse.ArgumentParser(description="Set the number of samples to test with")
    parser.add_argument("--address", help="Number of address records to sample")
    parser.add_argument("--author", help="Number of author records to sample")

    args = parser.parse_args()

    add_sample = None
    auth_sample = None
    if args.address:
        add_sample = args.address
    if args.author:
        auth_sample = args.author
    
    author_data, address_data = LoadData(add_sample, auth_sample)
    
    GoldenRecordCreation(author_data)