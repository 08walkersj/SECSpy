def validinput(inputstr, positive_answer, negative_answer):
    answer= input(inputstr+'\n')
    if answer==positive_answer:
        return True
    elif answer== negative_answer:
        return False
    else:
        print('Invalid response should be either '+ str(positive_answer)+ ' or ' +str(negative_answer))
        return validinput(inputstr, positive_answer, negative_answer)
