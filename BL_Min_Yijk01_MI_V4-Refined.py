#!/usr/bin/env python
# coding: utf-8

# In[36]:


"""
Overall Backlog Minimization - LP Model:

This linear optimization model tries to minimize the sum of the weighted backlog of all products. 

Input:
Demand Rate 
Production Rate
Yield of line/product
Minimum Lot Size
Initial Inventory Level
Target Inventory Level/Range
Backlog cost/value (weight)
Sequence dependent changeover time between products
Maximum makespan/cycle time allowed.

Objective:
To minimize the overall weighted backlog
To ending inventory is in the target inventory range for all products

Decision Variables:
    Major decision variables:
y[i,j,k,0] -> Binary variable. 1 if there is a changeover from product i to j in the same loop, 0 o/w
y[i,j,k,1] -> Binary variable. 1 if there is a changeover from product i to j in the subsequent loop, 0 o/w
x[i,k] -> Production time of product i in loop k
    
    Supporting decision variables:
z[i,k] -> Binary variable. 1 if product i is produced in loop k, 0 o/w
W[k] -> Binary variable. 1 if loop k is used


Output:
Final Inventory level of all products {I_last}
Actual makespan or cycle time (t)
Production Schedule (x[i,k], y[i,j,k,1 or 0])

Observations:
1. The model does not take the first ever setup time into account - In a large scale model, it is negligible. 
Using dummy starting product could resolve this problem.

2. Yield rate has not been implemented. It can be used for setting m/c utiliation as well. 

3. Max lot size has not been implemented. 

For all three of the above, we have some solution which requires further analysis. 

"""


# In[2]:


import numpy as np
from gurobipy import *
import pandas as pd
import time


# In[3]:


m = Model("target_inventory")


# In[4]:


#Example Instance
demand = [5,2,8] #Demand Rate per hour
prod_rate = [28,23,35] #Production rate per hour
Y = [0.75,0.75,0.75] #yield for product i -> Not used
Min_lot = [20,20,20] #Minimum lot size
Max_lot = [1000000000,100000000,10000000] #Maximum lot size - not used
I_zero = [5,10,0] #Initial Inventory
I_target_low = [500,200,300] #Target Inventory - lower bound
I_target_upper = [2000,500,700] #Target Inventory - upper bound
bcost = [10,2,6] #backlog cost/value
C = np.array([[10000,5,4],[4,10000,5],[6,7,10000]]) #Changeover Time between products in hours
loops = 10 #No. of loops, this could be altered for different no. of changeovers. 
prods = [1,2,3] #product sequence or numbers. 
T = 70 #Maximum makespan or cycle time allowed


# In[5]:


#range of products and loops
nprod = range(len(prods)) #for products
nloops = range(loops) #for loops
nmloops = range(loops-1) #for subsequent loops -> not used anywhere
LI = [0,1] #loop indicator -> zero for same, and 1 for next loop 


# # Variable Declaration

# In[6]:


y = m.addVars(nprod,nprod,nloops,LI, vtype = GRB.BINARY, name = "y") #to indicate switch from product i to product j
z = m.addVars(nprod,nloops, vtype = GRB.BINARY, name = "z") #to indicate if m/c has to be setup for product i in loop k
x = m.addVars(nprod,nloops, vtype = GRB.INTEGER, lb = 0, name = "x") #production time of product i in loop k
inv_k = m.addVars(nprod,nloops, vtype = GRB.CONTINUOUS, lb = -10000000, name = "inv_k") #inventory at the beginning of loop k
#rows are products and columns are loops
L = m.addVars(nprod,nloops, vtype = GRB.CONTINUOUS,lb = -100000000, name = "L") # maximum backlog
#T = m.addVar(vtype=GRB.CONTINUOUS, name = "total_time")
W = m.addVars(nloops, vtype = GRB.BINARY, name = "W")#"Continuous_loop_check")
I_last = m.addVars(nprod, vtype = GRB.CONTINUOUS, name = "I_final")
t = m.addVar(vtype =GRB.CONTINUOUS, name = "cycle time")


# # Objective Function

# In[7]:


#Objective Function: Weighted backlog 1|s_{jk}|\sum_{i \in [n]}b_{i}L_{i} 
m.setObjective(sum(bcost[i]*L[i,k] for i in nprod for k in nloops), GRB.MINIMIZE)  #+T removed 


# # Constraints

# In[8]:


#Constraints for backlog Li

#backlog should be greater than 0
m.addConstrs(L[i,k] >= 0 for i in nprod for k in nloops) 

#backlog should be greater than the negative of negative inventory so that it forces it be a positive
m.addConstrs(L[i,k] >= -inv_k[i, k] for i in nprod for k in nloops);


# In[9]:


#constraint for initial inventory at beginning of first loop
m.addConstrs(inv_k[i,0] == I_zero[i] 
             - demand[i] *sum(x[j,0] for j in nprod  if j < i)
             - demand[i]*sum(C[l,j]*y[l,j,0,0] for j in nprod for l in nprod if l<j if j<=i) for i in nprod);


# In[10]:


#inventory constraint for every loop for i in nprod  
#(loops - 1) is to prevent from going out of range
m.addConstrs(inv_k[i, k+1] == inv_k[i,k] + x[i,k]*(prod_rate[i] - demand[i]) - 
             demand[i]*(((sum(C[l,j]*y[l,j,k,0] for l in nprod if l>=i for j in nprod if j>l) 
                             + sum(C[l,j]*y[l,j,k,1] for j in nprod if j<=i for l in nprod if l>=i))) 
                        + sum(C[l,j]*y[l,j,k+1,0] for j in nprod if i>=j for l in nprod  if j>l))
             for i in nprod for k in range(loops-1));


# In[11]:


#constraint for target inventory - lower bound
k_last = loops - 1
m.addConstrs(I_target_low[i] <= inv_k[i,k_last] +  x[i,k_last]*(prod_rate[i] - demand[i])
             - demand[i] * sum(x[j,k_last] for j in nprod if j > i) 
             - demand[i]*sum(C[l,j]*y[l,j,k_last,0] for l in nprod if l>=i for j in nprod if j>l) for i in nprod);


# In[12]:


#constraint for target inventory - upper bound
k_last = loops - 1
m.addConstrs(I_target_upper[i] >= inv_k[i,k_last] +  x[i,k_last]*(prod_rate[i] - demand[i])
             - demand[i] * sum(x[j,k_last] for j in nprod if j > i) 
             - demand[i]*sum(C[l,j]*y[l,j,k_last,0] for l in nprod if l>=i for j in nprod if j>l) for i in nprod);


# In[13]:


#Final inventory level - the last loop
m.addConstrs(I_last[i] == inv_k[i,k_last] +  x[i,k_last]*(prod_rate[i] - demand[i])
             - demand[i] * sum(x[j,k_last] for j in nprod if j > i) 
             - demand[i]*sum(C[l,j]*y[l,j,k_last,0] for l in nprod if l>=i for j in nprod if j>l) for i in nprod);


# In[14]:


#Constraint to ensure that the makespan is less than the desired cycle time T
m.addConstr(T >= sum(x[i,k] for i in nprod for k in nloops) + sum(C[i,j]*y[i,j,k,u] for i in nprod for j in nprod for k in nloops for u in LI));

#Constraint to record the actual makespan or cycle time 
m.addConstr(t == sum(x[i,k] for i in nprod for k in nloops) + sum(C[i,j]*y[i,j,k,u] for i in nprod for j in nprod for k in nloops for u in LI));


# In[15]:


#Constraint to record if product i has been produced in loop k through binary variable z
m.addConstrs(z[i,k] >= x[i,k]/T for i in nprod for k in nloops) 


# In[16]:


#constraint to set the values of yij for all products where j > i IN THE SAME LOOP

#m.addConstrs(y[i,j,k,0] >= z[i,k] + z[j,k] - 1 - sum(z[u,k] for u in range(i+1,j-1) if j>i) for j in nprod for i in nprod for k in nloops if j>i);

#constraint to set the values of yij for all products where j > i IN THE SAME LOOP
#m.addConstrs(y[i,j,k,0] >= z[i,k] + z[j,k] - 1 - sum(z[u,k] for u in range(i+1,j) if j>i) for j in nprod for i in nprod for k in nloops if j>i);

#Below Changes added on 11/14/19
#Constraint to set the values of y[ijk0] for all products where j > i, in the same loop
m.addConstrs(y[i,j,k,0] >= z[i,k] + z[j,k] - 1 - sum(z[u,k] for u in range(i+1,j) if j>i if j!=i+1) for j in nprod for i in nprod for k in nloops if j>i);

m.addConstrs(y[i,j,k,0] >= z[i,k] + z[j,k] - 1 for j in nprod for i in nprod for k in nloops if j>i if j==i+1);


# In[17]:


#constraint to set the values of y[ijk1] for all products where j < i, in subsequent loops
m.addConstrs(y[i,j,k,1] >= z[i,k] + z[j,k+1] - 1 - sum(z[u,k] for u in nprod if u>i) - sum(z[v,k+1] for v in nprod if v<j)
                for j in nprod for i in nprod for k in range(loops-1));


# In[18]:


#Constraint to ensure that the changeover between products in subsequent loops
#happen from higher index to lower index only.
m.addConstrs(y[i,j,k,1] == 0 for i in nprod for j in nprod if j>i for k in range(loops-1) );

#Observation: Necessary constraint otherwise y[0,1,0,0] and y[0,1,0,1] are being set to 1, which cannot happen. 


# In[19]:


#Sharat -> 5/12/19, Constraint to avoid changeover between the same product between different loops
m.addConstrs(y[i,i,k,0] == 0 for i in nprod for k in nloops );
m.addConstrs(y[i,i,k,1] == 0 for i in nprod for k in range(loops-1) );


# In[20]:


#Constraints to ensure changeover from product i to any other product j!=i happens only once in the same loop or between subsequent loop
m.addConstrs(quicksum(y[i,j,k,0] for j in nprod if j>i) <= 1 for i in nprod for k in nloops)
m.addConstrs(quicksum(y[i,j,k,1] for j in nprod if j<i) <= 1 for i in nprod for k in range(loops-1))

#Ensure y[ijku] is 1 or 0 when it is expected to be that value. This is necessary as it does not allow the system to play around the
#values of these variables.
m.addConstrs(y[i,j,k,0] <= (z[i,k] + z[j,k] + 0.9)/2 for i in nprod for j in nprod for k in nloops if j>i)
m.addConstrs(y[i,j,k,1] <= (z[i,k] + z[j,k+1] + 0.9)/2 for i in nprod for j in nprod for k in range(loops-1) if j<i)

#Ensures z is indeed zero when it is expected to be zero.This is necessary as it does not allow the system to play around the
#values of these variables.
m.addConstrs(z[i,k] <= x[i,k]*T for i in nprod for k in nloops)  #Could lead to infeasibility (any product when manufacutred must be for min of few minutes.)

#Observation: The constraint above "z[i,k] <= x[i,k]*T" requires 1>=x[i,k] or x[i,k]=0. O/w it could make the system infeasible. 


# In[21]:


#Constraints to ensure that a loop in between is not skipped

m.addConstr(W[0] == 1); #Ensures loop 0 is definitely used
m.addConstrs(W[k] >= z[i,k] for i in nprod for k in nloops); #Pushed W to be 1 if any product i is produced in loop k
m.addConstrs(W[k] <= sum(z[i,k] for i in nprod) for k in nloops); #Ensures W is indeed zero when we expect it to be zero
m.addConstrs(W[k+1] <= W[k] for k in range(loops-1)); # loop k+1 cannot be used unless k is used


# In[22]:


#Constraints to ensure that when a product i is produced, it is produced for at least the given minimum lot size.
m.addConstrs(x[i,k]*prod_rate[i] >= z[i,k]*Min_lot[i] for i in nprod for k in nloops)


# # Solving the model

# In[23]:


start_time = time.time() #Optimization begins
m.optimize();             
end_time = time.time() #Optimization ends


# In[24]:


#Code snippet to retrieve solution
sol = []
for v in m.getVars():
    sol.append((v.Varname, v.X))
    #print("%s %f" % (v.Varname, v.X))
print("The optimal obj. value: "+str(m.objVal))    
#m.write("Backlog_minimization.sol");


# In[25]:


print("Run-time of the model: "+str(end_time-start_time)+" sec")


# In[26]:


#Added by Sharat on 10/18/19 to retrieve only the activated changeover binary variable.

solx=[]
solx= m.getAttr('x', x)   
soly=[]
soly= m.getAttr('x', y)   
solz=[]
solz= m.getAttr('x', z)   

Final_soly =[] 
Sequence = []
proctime = []

for k in nloops:
    for i in nprod:
        for j in nprod:
            for L in range(0,2):
                if soly[i,j,k,L] == 1:
                    Final_soly.append([i,j,k,L])
                    

Final_soly
#Sequence
#proctime
#print("time taken = "+str(end_time-start_time))


# In[27]:


for k in nloops:
    for i in nprod:
        if solz[i,k] ==1:
            print("z("+str(i)+","+str(k)+")"+" = "+str(solz[i,k]))


# In[28]:


for k in nloops:
    for i in nprod:
        if solx[i,k] > 0:
            #print("processing time of product "+str(i)+" in loop "+str(k)+" = "+str(solx[i,k]))
            print("x("+str(i)+","+str(k)+")"+" = "+str(solx[i,k]))
                    
        
                


# In[29]:


I_last


# In[30]:


t


# In[31]:


df = pd.DataFrame(sol)
df.to_csv("Solutions_formulation_new.csv", header = ["Name","Value"], index = False)


# In[32]:


inv_val = [[prod, round(x[prod,loop].X,2),round(inv_k[prod,loop].X,2)] for prod in nprod for loop in nloops]
inv_val = np.array(inv_val)


# In[33]:


inv_val


# In[34]:


import matplotlib.pyplot as plt


# In[35]:


for item in range(len(inv_val)):
    if inv_val[item,0] == 0:
        print(inv_val[item,1], inv_val[item,2])
        plt.plot(inv_val[item,1], inv_val[item,2], 'ro')
plt.show()

