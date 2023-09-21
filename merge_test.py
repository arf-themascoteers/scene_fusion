import pandas as pd


d1 = pd.DataFrame(data=
             [
                 [1,2,3],
                 [2,5,6],
                 [3,8,9]
              ],
             columns = ["a","b","c"]
             )

d2 = pd.DataFrame(data=
             [
                 [1,20,30],
                 [2,50,60],
                 [3,80,90]
              ],
             columns = ["a","b","c"]
             )

d3 = pd.concat((d1, d2), axis=0)
print(d3.index)
d3 = d3.groupby(d3.index).mean().reset_index(drop=True)
print(d3)