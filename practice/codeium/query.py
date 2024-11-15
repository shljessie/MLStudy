document = [the, fox, dog,* ,the, cat]
query = [fox, dog,* ,the] # true
query = [fox, cat] # false
query = [the, cat] #true


# fist make the document into a hashmap cus we have to do a search of the query values from the document
# easy lookup we need to do a hashmap

# words have to be consecutive indexes from eachother 
# 