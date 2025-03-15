from fusionrag import evaluation
import pandas as pd

testing_queries = pd.read_csv('queries_output.csv', encoding='utf-8')

queries = []
for i in range(len(testing_queries)):
    queries.append(testing_queries.loc[i, "Expert_Legal"])
    queries.append(testing_queries.loc[i, "Professional_Legal"])
    queries.append(testing_queries.loc[i, "Informed_Layperson"])
    queries.append(testing_queries.loc[i, "General_Public"])


evaluation(queries)