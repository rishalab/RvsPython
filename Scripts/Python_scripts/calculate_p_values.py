
import scipy.stats as stats
 #### Directly https://www.socscistatistics.com/tests/signedranks/default2.aspx can be used to obtain p values


#For example, put the pyJoules output for a particular task
group1 = [5.09, 2.94, 2.72, 5.82, 3.65, 4.34, 3.43, 4.02, 2.82, 3.48] 
#put the RJoules output for the same task
group2 = [65, 87, 456, 564, 456, 564, 564, 6, 4, 564]
 
# conduct the Wilcoxon-Signed Rank Test
P=stats.wilcoxon(group1, group2)
print(P)
        
