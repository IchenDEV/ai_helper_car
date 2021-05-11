def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue

if __name__ == '__main__':
    inferred_request = [0]*4
    print(inferred_request)
    print(searchlist(inferred_request,0))