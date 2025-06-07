printlist={}
for i in range (3):
    unique_id=i
    print(unique_id)
    if unique_id not in printlist.keys():
                        printlist.update({unique_id:['y',-1,-1]})
print(printlist)