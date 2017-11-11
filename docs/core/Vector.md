The Vector object stores data in a pandas DataFrame and provides support for performing calculations on this data.

#####Example
```
index=['a','b','c']
data=[1,2,3]
v=Vector(index=index, data=data, columns='some_random_data')
```
This will create a Vector instance whose df attribute is a 3x1 DataFrame containing the elements of `data` and rows indexed by `index`. The column names (in this case, one column name) is defined by `columns`.

`data` and `index` have the same format as they would if they were passed to the initialization function of a DataFrame directly. For more information, see the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

####Vector().partial_kl(q, which, column=None)
Return the partial KL of the observation in this instance&#39;s df attribute at row `which` and column `column` with the observation in q&#39;s (a Vector instance) at row `which` and column `column`.
