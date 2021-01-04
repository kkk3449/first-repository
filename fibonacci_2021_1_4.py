#!/usr/bin/env python
# coding: utf-8

# In[1]:


memory ={1:1, 2:1}

def fibonacci(n) :
    if n in memory:
        number = memory[n]
    else :
        number = fibonacci(n-1)+fibonacci(n-2)
        memory[n] = number
    return number

print(fibonacci(100))

print(memory)
    


# In[ ]:




