import pickle, reprlib
p='e:/whatsapp agent poc/faiss_index/index.pkl'
with open(p,'rb') as f:
    obj=pickle.load(f)
print(type(obj))
if isinstance(obj,list):
    print('len', len(obj), 'type', type(obj[0]))
elif isinstance(obj,dict):
    print('keys', list(obj.keys()))
    for k in list(obj.keys())[:5]:
        print(k, type(obj[k]))
else:
    print(reprlib.repr(obj)[:1000])
