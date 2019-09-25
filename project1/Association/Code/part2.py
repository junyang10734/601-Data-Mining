from itertools import combinations

dataList = {}


# load dataset
def loadDataSet():
    with open('associationruletestdata.txt', 'r') as f:
        data = f.readlines()
        dataList = []
        for i in range(len(data)):
            item = data[i].split('\t')
            for idx in range(len(item)-1):
                item[idx] = 'G' + str(idx+1) + '_' + item[idx]  # Adding Prefixes to Gene Expression
            dataList.append(item)
    return dataList  # return a list


# Generates itemsets of length 1
# dataList: loaded dataset
def creatSet1(dataList):
    d = {}
    # scanning each item and storing items in the into dictionary d
    # if an item is not in the dictionary, then add to the dictionary;
    # otherwise, the count of this item is incremented by 1
    for transction in dataList:
        for item in transction:
            if item in d:
                d[item] = d[item] + 1
            else:
                d[item] = 1
    return d


# generate candidate itemsets
# Fk-1 x Fk-1 Method
# this operation generates new candidate k-itemsets based on the frequent (k-1)-itemsets
# merged two (k-1)-itemsets if their first k-2 items are same
# k: the length of each itemsets we want to generate
# dicSet: the dictionary which stores calculated frequent itemsets
def apriori_gen(k, dicSet):
    set0 = dicSet[k-1]  # get length-(k-1) frequent itemsets
    setk = {}  # store the calculated length-k itemsets

    if k == 2:  # when k==2, it can be combined directly from length-1 itemsets without comparing the first k-2 attributes
        attrs = []
        for key in set0:
            attrs.append(key)

        i = 0
        while( i< len(attrs)-1 ):  # scanning length-1 itemsets
            j = i+1
            while (j < len(attrs)):
                item = attrs[i] + ' ' + attrs[j]  # combined 2 items to 1 item
                setk[item] = 0  # adds a new length-2 itemset to setk with an initial count of 0
                j = j + 1
            i = i + 1
    else:     # when k!=2, compare the first k-2 attributes of each itemset
        attrsList = []
        for key in set0:
            attrsList.append(key)

        i = 0
        while (i < len(attrsList) - 1):   # scan length-(k-1) itemsets
            j = i + 1
            while (j < len(attrsList)):
                arr1 = attrsList[i].split()  # list of attributes for the first itemset
                arr2 = attrsList[j].split()  # list of attributes for the second itemset
                if arr1[:k - 2] == arr2[:k - 2]:   # compare the first k-2 attributes of the two itemsets, if they are the same, merging to generate a new itemset with length-k
                    item = arr1
                    item.append(arr2[-1])
                    setk[' '.join(item)] = 0  # convert new attributs list into an attribute string and store it in a dictionary with an initial count of 0
                j = j + 1
            i = i + 1
    return setk


# Calculate the support of each itemset
def supCount(setk):
    for key in setk:
        keyList = set(key.split())
        for transaction in dataList:
            if keyList.issubset(
                    set(transaction)):  # if an itemset exists in the datalist, its count is incremented by 1.
                setk[key] = setk[key] + 1
    return setk


# prune infrequency itemsets and return the trimmed dictionary
# times: min-support
# d: itemsets which we want to prune
def prune(times, d):
    delList = []
    for key, value in d.items():
        if(value < times):  # if the count of an itemset is less than min-support, store it in delList
            delList.append(key)
    for i in delList:  # delete infrequency itemsets
        del d[i]
    return d


class Rule:
    def __init__(self, rule, head, body): # Set class attributes
        self.rule = rule
        self.head = head
        self.body = body

    def toString(self):
        return 'rule'+ str(self.head) + '->' + str(self.body) # return a visible rule

    def getRule(self):
        return self.rule # Return rules

    def getHead(self):
        return self.head # Return heads

    def getBody(self):
        return self.body # Return bodies

def rulePrune(rule, head, conf): # Determine whether rules meet the confidence need
    if (rule/head < conf/100): # If the confidence less than the confidence margin
        # print('fail ' + str(rule) + " / " + str(head) + " = " + str(rule/head))
        return 0
    else:
        # print('success ' + str(rule) + " / " + str(head) + " = " + str(rule/head))
        return 1

def rulePair(dk, dicSet, conf):
    k = 0
    attrsList1 = []
    rules = []
    for key in dk:
        attrsList1.append(key) # AL1 stores the keys in dk
    attrsList2 = []
    for h in range(len(dicSet)):
        for key in dicSet[h]:
            attrsList2.append(key) # AL2 stores the keys for earlier frequent sets

    for i in range(len(attrsList1)):
        for j in range(len(attrsList2)):
            att1 = attrsList1[i].split() # Split genes
            att2 = attrsList2[j].split()
            # print('att1 = ' + str(att1))
            # print('att2 = ' + str(att2))
            for l in att1:
                for m in att2:
                    if (l == m):
                        att1.remove(l) # Remove them from the compairing sets
                        att2.remove(m)
            for l in att1:
                for m in att2:
                    if (l == m):
                        att1.remove(l) # Remove them from the compairing sets
                        att2.remove(m)
            if (len(att2) != 0): # We want att2 empty to make sure every item in att2 in att1
                continue
            # print(attrsList2[j])
            if rulePrune(dk[attrsList1[i]], dicSet[len(attrsList2[j].split())][attrsList2[j]], conf): # Remove rules don't meet the cofidence need
                rule = Rule(attrsList1[i].split(), attrsList2[j].split(), att1) # Store them as an object
                print("rule " + str(k) + " is " + rule.toString())
                k += 1
                rules.append(rule) # append rules objects to the list
    return rules

def getRuleWithType(ruleSet, ruleType): #Return with rule/head/body wanted
    ds = []
    for rule in ruleSet:
        if ruleType == "RULE":
            ds.append(rule.getRule()) # Get rules
        if ruleType == "HEAD":
            ds.append(rule.getHead()) # Get heads
        if ruleType == "BODY":
            ds.append(rule.getBody()) # Get bodies
    return ds

def template1(ruleSet, ruleType, condition, gene):
    ds = getRuleWithType(ruleSet, ruleType) # Get rule/head/body from rules
    list = [] # Store the line number of quary results
    if (condition == "ANY"):
        for index in range(len(ds)): # for each attribute in datasets
            for d in ds[index]: # For each gene in attribute
                if (gene[0] == d):
                    list.append(index) # Record the line number
                    break
                if (len(gene) == 2): # If gene has length of two
                    if (gene[1] == d): # If first not match, try to match the second one
                        list.append(index) # Record the line number
                        break  # Go for next loop
    if (condition == 1):
        for index in range(len(ds)):  # for each attribute in datasets
            if (gene[0] in ds[index]):  # First compaire the first gene
                list.append(index)  # Record the line number
            if (len(gene) == 2):  # If gene has length of two
                if (gene[1] in ds[index]):  # If first not match, try to match the second one
                    list.append(index)  # Record the line number
                if (gene[0] in ds[index] and gene[1] in ds[index]):
                    list.remove(index)
    if (condition == "NONE"):
        for index in range(len(ds)): # for each attribute in datasets
            exist = 0
            for d in ds[index]: # For each gene in attribute
                if (gene[0] == d): # If matches
                    exist = 1 # Make next loop
                    break  # Go for next loop
                if (len(gene) == 2): # If gene has length of two
                    if (gene[1] == d): # If first not match, try to match the second one
                        exist = 1  # Make next loop
                        break # Go for next loop
            if(exist):
                continue
            list.append(index) # Record the line number
    return list

def template2(ruleSet, ruleType, number):
    ds = getRuleWithType(ruleSet, ruleType) # Get rule/head/body from rules
    list = [] # Store the line number of quary results
    for index in range(len(ds)): # for each attribute in datasets
        if len(ds[index]) >= number: # Length of rule/head/body >= number
            list.append(index) # Record the line number
    return list

def template3(ruleSet, condition, *param):
    if condition == "1or1":
        list1 = template1(ruleSet, param[0], param[1], param[2]) # Get results
        list2 = template1(ruleSet, param[3], param[4], param[5])
        for line in list1: # Combine results
            if line not in list2:
                list2.append(line)
        return list2
    if condition == "1and1":
        list1 = template1(ruleSet, param[0], param[1], param[2]) # Get results
        list2 = template1(ruleSet, param[3], param[4], param[5])
        list3 = [] # Union results
        for line in list1:
            if line in list2:
                list3.append(line)
        return list3
    if condition == "1or2":
        list1 = template1(ruleSet, param[0], param[1], param[2]) # Get results
        list2 = template2(ruleSet, param[3], param[4])
        for line in list1: # Combine results
            if line not in list2:
                list2.append(line)
        return list2
    if condition == "1and2":
        list1 = template1(ruleSet, param[0], param[1], param[2]) # Get results
        list2 = template2(ruleSet, param[3], param[4])
        list3 = [] # Union results
        for line in list1:
            if line in list2:
                list3.append(line)
        return list3
    if condition == "2or2":
        list1 = template2(ruleSet, param[0], param[1]) # Get results
        list2 = template2(ruleSet, param[2], param[3])
        for line in list1: # Combine results
            if line not in list2:
                list2.append(line)
        return list2
    if condition == "2and2":
        list1 = template2(ruleSet, param[0], param[1]) # Get results
        list2 = template2(ruleSet, param[2], param[3])
        list3 = [] # Union results
        for line in list1:
            if line in list2:
                list3.append(line)
        return list3


def main():
    f = open('answer.txt', 'w')
    global dataList
    dataList = loadDataSet()
    tnum = len(dataList)  # total number of transactions
    minsup = 50  # set minium support
    minConf = 70  # Minium Confidence
    dicSet = [{}]  # store frequent itemsets
    ruleSet = []
    d1 = creatSet1(dataList)  # generate length-1 candidate itemsets
    d1 = prune(int((minsup/100)*tnum), d1)  # prune itemsets, generate frequent length-1 itemsets
    dicSet.append(d1)
    f.write('number of length-1 frequent itemsets: ' + str(len(d1)))
    f.write('\n' + str(d1))
    sum = len(d1)
    k = 2
    dk = d1
    while (len(dk)>0):  # start from k==2, generate new frequent itemsets and rules
        dk = apriori_gen(k, dicSet)
        dk = supCount(dk)
        dk = prune(int((minsup/100)*tnum), dk)
        rk = rulePair(dk, dicSet, minConf)  # Get association rules with over 70% confidence
        dicSet.append(dk)  # store frequent itemsets into dicSet
        for rule in rk:
            ruleSet.append(rule)  # Store rules into ruleSet
        if len(dk)!= 0:
            f.write('\nnumber of length-' + str(k) + ' frequent itemsets: ' + str(len(dk)))
            f.write('\n' + str(dk))
            sum = sum + len(dk)
            k = k + 1
        # f.write('\nnumber of length-' + str(k) + ' frequent itemsets:' + str(len(dk)))
        # f.write('\n' + str(dk))
        # sum = sum + len(dk)
        # k = k + 1
    f.write('\nTotal number of frequent itemsets: ' + str(sum))
    f.write("\nnumber of rules is " + str(len(ruleSet)))


    f.write('\n\nTemplate1')
    f.write('\nasso_rule.template1("RULE", "ANY", [\'G59_UP\'])\t\t\t' + str(len(template1(ruleSet, "RULE", "ANY", ['G59_Up']))))
    f.write('\nasso_rule.template1("RULE", "NONE", [\'G59_UP\']) \t\t' + str(len(template1(ruleSet, "RULE", "NONE", ['G59_Up']))))
    f.write('\nasso_rule.template1("RULE", 1, [\'G59_UP\', \'G10_Down\'])\t' + str(len(template1(ruleSet, "RULE", 1, ['G59_Up', 'G10_Down']))))
    f.write('\nasso_rule.template1("HEAD", "ANY", [\'G59_UP\'])\t\t\t' + str(len(template1(ruleSet, "HEAD", "ANY", ['G59_Up']))))
    f.write('\nasso_rule.template1("HEAD", "NONE", [\'G59_UP\'])\t\t\t' + str(len(template1(ruleSet, "HEAD", "NONE", ['G59_Up']))))
    f.write('\nasso_rule.template1("HEAD", 1, [\'G59_UP\', \'G10_Down\'])\t' + str(len(template1(ruleSet, "HEAD", 1, ['G59_Up', 'G10_Down']))))
    f.write('\nasso_rule.template1("BODY", "ANY", [\'G59_UP\'])\t\t\t' + str(len(template1(ruleSet, "BODY", "ANY", ['G59_Up']))))
    f.write('\nasso_rule.template1("BODY", "NONE", [\'G59_UP\'])\t\t\t' + str(len(template1(ruleSet, "BODY", "NONE", ['G59_Up']))))
    f.write('\nasso_rule.template1("BODY", 1, [\'G59_UP\', \'G10_Down\'])\t' + str(len(template1(ruleSet, "BODY", 1, ['G59_Up', 'G10_Down']))))

    f.write('\nTemplate2')
    f.write('\nasso_rule.template2("RULE", 3)\t' + str(len(template2(ruleSet, "RULE", 3))))
    f.write('\nasso_rule.template2("HEAD", 2)\t' + str(len(template2(ruleSet, "HEAD", 2))))
    f.write('\nasso_rule.template2("BODY", 1)\t' + str(len(template2(ruleSet, "BODY", 1))))

    f.write('\nTemplate3')
    f.write('\nasso_rule.template3("1or1", "HEAD", "ANY", [\'G10_Down\'], "BODY", 1, [\'G59_UP\'])\t\t' + str(len(template3(ruleSet, "1or1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up']))))
    f.write('\nasso_rule.template3("1and1", "HEAD", "ANY", [\'G10_Down\'], "BODY", 1, [\'G59_UP\'])\t' + str(len(template3(ruleSet, "1and1", "HEAD", "ANY", ['G10_Down'], "BODY", 1, ['G59_Up']))))
    f.write('\nasso_rule.template3("1or2", "HEAD", "ANY", [\'G10_Down\'], "BODY", 2)\t\t\t\t\t' + str(len(template3(ruleSet, "1or2", "HEAD", "ANY", ['G10_Down'], "BODY", 2))))
    f.write('\nasso_rule.template3("1and2", "HEAD", "ANY", [\'G10_Down\'], "BODY", 2)\t\t\t\t' + str(len(template3(ruleSet, "1and2", "HEAD", "ANY", ['G10_Down'], "BODY", 2))))
    f.write('\nasso_rule.template3("2or2", "HEAD", 1, "BODY", 2)\t\t\t\t\t\t\t\t\t'+str(len(template3(ruleSet, "2or2", "HEAD", 1, "BODY", 2))))
    f.write('\nasso_rule.template3("2and2", "HEAD", 1, "BODY", 2)\t\t\t\t\t\t\t\t\t' + str(len(template3(ruleSet, "2and2", "HEAD", 1, "BODY", 2))))

    f.close()

main()
