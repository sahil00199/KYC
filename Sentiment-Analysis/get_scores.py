import os
import sys

def startsWith(base, searchString):
    if len(base) > len(searchString) and base[:len(searchString)] == searchString: return True
    return False

def getBestVal(directory):
    bestYet = -1
    bestFile = "eval_results.txt"
    for filename in os.listdir(directory):
        if startsWith(filename, "eval_results_ep"):
            file = open(os.path.join(directory, filename))
            lines = [x.strip() for x in file.readlines()]
            lines = [x for x in lines if startsWith(x, "eval_accuracy")]
            file.close()
            scores = [float(line.split(' ')[-1]) for line in lines]
            if scores[0] > bestYet:
                bestYet = scores[0]
                bestFile = filename
    return bestFile

def getAcc(directory):
    eval_file_name = getBestVal(directory)
    file = open(os.path.join(directory, eval_file_name))
    lines = [x.strip() for x in file.readlines()]
    lines = [x for x in lines if len(x) > 13 and x[:13] == 'eval_accuracy']
    file.close()
    lines = [float(line.split(' ')[-1]) for line in lines]
    return lines, eval_file_name[len("eval_results_ep_")]

for file in ['vanilla', 'kyc']:
    print('*' * 20, file.upper(), '*' * 20)
    file = os.path.join('models', file)
    scores, epochNumber = getAcc(file)
    print("Best Validation Score after epoch", epochNumber)
    print('Dev:', str(scores[0]))
    print('Test:', str(scores[1]))
    print('OOD Test:', str(scores[2]))
    print()


