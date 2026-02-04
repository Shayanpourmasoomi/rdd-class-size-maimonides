```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
```

# Q_11



```python
data = pd.read_stata('maimonides.dta')

```


```python

vars = ['classize', 'avgmath', 'avgverb', 'enrollment', 'perc_disadvantaged']

missing_vars = [var for var in vars if var not in data.columns]
if missing_vars:
    print(f" {missing_vars}")
else:
    desc_stats = data[vars].describe()

    desc_stats.to_csv('descriptive_stats.csv')

    print(" Summary stat    :")
    print(desc_stats.to_string())


```

     Summary stat    :
              classize      avgmath      avgverb   enrollment  perc_disadvantaged
    count  2029.000000  2024.000000  2024.000000  2029.000000         2029.000000
    mean     29.954165    67.323006    74.445480    77.868408           14.091178
    std       6.598022    10.032690     8.077352    39.060742           13.484891
    min       5.000000     0.000000    34.799999     5.000000            0.000000
    25%      26.000000    61.137501    69.857498    50.000000            4.000000
    50%      31.000000    67.800003    75.430000    72.000000           10.000000
    75%      35.000000    74.095001    79.847000   100.000000           19.000000
    max      47.000000   181.246002   187.606003   226.000000           76.000000
    

# Q_12



```python
data = data[(data['avgmath'] <= 100) & (data['avgmath'] > 0)].copy()

plt.figure(figsize=(10, 6))

sns.scatterplot(x='enrollment', y='classize', data=data, alpha=0.5, label='Observation')

thresholds = [41, 81, 121, 161, 201]
for thresh in thresholds:
    plt.axvline(x=thresh, color='red', linestyle='--', alpha=0.7, label='Cutoff' if thresh == 41 else "")

bins = np.arange(0, data['enrollment'].max() + 10, 10)     
data['enrollment_bin'] = pd.cut(data['enrollment'], bins, labels=bins[:-1])
bin_means = data.groupby('enrollment_bin', observed=True)['classize'].mean().reset_index()

plt.plot(bin_means['enrollment_bin'], bin_means['classize'], marker='o', color='blue', label=' Local Average  ')

plt.title('Average class size by number of enrollments')
plt.xlabel('(enrollment)')
plt.ylabel('  (classize)')
plt.legend()
plt.grid(True)

plt.savefig('classize_vs_enrollment.png')
#plt.show()
```


    
![png](output_5_0.png)
    


#  Q_13


```python

model1 = smf.ols('avgmath ~ classize + perc_disadvantaged', data=data).fit()

model2 = smf.ols('avgmath ~ classize + perc_disadvantaged + enrollment', data=data).fit()

data_subset = data[(data['enrollment'] >= 20) & (data['enrollment'] <= 60)].copy()
model3 = smf.ols('avgmath ~ classize + perc_disadvantaged + enrollment', data=data_subset).fit()

print("Model 1 (similar to column 4):")
print(model1.summary().tables[1])
print("\nModel 2 (similar to column 5):")
print(model2.summary().tables[1])
print("\nModel 3 (similar to column 6, restricted to enrollment between 20 and 60):")
print(model3.summary().tables[1])

with open('ols_results.txt', 'w', encoding='utf-8') as f:
    f.write("Model 1 (similar to column 4):\n")
    f.write(model1.summary().as_text())
    f.write("\nModel 2 (similar to column 5):\n")
    f.write(model2.summary().as_text())
    f.write("\nModel 3 (similar to column 6):\n")
    f.write(model3.summary().as_text())
```

    Model 1 (similar to column 4):
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             69.9032      1.008     69.370      0.000      67.927      71.879
    classize               0.0731      0.030      2.437      0.015       0.014       0.132
    perc_disadvantaged    -0.3400      0.015    -23.186      0.000      -0.369      -0.311
    ======================================================================================
    
    Model 2 (similar to column 5):
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             70.1487      1.010     69.448      0.000      68.168      72.130
    classize               0.0175      0.036      0.483      0.629      -0.054       0.089
    perc_disadvantaged    -0.3323      0.015    -22.273      0.000      -0.362      -0.303
    enrollment             0.0168      0.006      2.729      0.006       0.005       0.029
    ======================================================================================
    
    Model 3 (similar to column 6, restricted to enrollment between 20 and 60):
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             66.9999      2.050     32.689      0.000      62.976      71.024
    classize               0.1313      0.066      1.997      0.046       0.002       0.260
    perc_disadvantaged    -0.3323      0.020    -16.863      0.000      -0.371      -0.294
    enrollment             0.0234      0.029      0.821      0.412      -0.033       0.079
    ======================================================================================
    

# Q_14


```python

data['large_class'] = (data['enrollment'] >= 41).astype(int)

data_a = data[(data['enrollment'] >= 20) & (data['enrollment'] <= 60)].copy()

data_a['enrollment_centered'] = data_a['enrollment'] - 41

model_a = smf.ols('avgmath ~ large_class + enrollment_centered + perc_disadvantaged', data=data_a).fit()

print("RDD with Cutoff 41")
print(model_a.summary().tables[1])

data['predicted_size'] = data['enrollment'] / (np.floor((data['enrollment'] - 1) / 40) + 1)

model_b = smf.ols('avgmath ~ predicted_size + perc_disadvantaged', data=data).fit()

print("\n RDD with all Cutoff")
print(model_b.summary().tables[1])

# ذخیره نتایج در فایل با رمزگذاری UTF-8
with open('rdd_results.txt', 'w', encoding='utf-8') as f:
    f.write("Section (a): Sharp RDD at threshold 41\n")
    f.write(model_a.summary().as_text())
    f.write("\nSection (b): Sharp RDD with all thresholds (similar to column 6, Table III)\n")
    f.write(model_b.summary().as_text())
```

    RDD with Cutoff 41
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              69.4308      0.945     73.493      0.000      67.576      71.286
    large_class             3.4231      1.366      2.505      0.012       0.740       6.106
    enrollment_centered    -0.0857      0.055     -1.557      0.120      -0.194       0.022
    perc_disadvantaged     -0.3395      0.019    -17.415      0.000      -0.378      -0.301
    =======================================================================================
    
     RDD with all Cutoff
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             72.6827      1.075     67.609      0.000      70.574      74.791
    predicted_size        -0.0126      0.032     -0.396      0.692      -0.075       0.050
    perc_disadvantaged    -0.3543      0.014    -24.621      0.000      -0.383      -0.326
    ======================================================================================
    

# Q_15


```python

data['above_41'] = (data['enrollment'] >= 41).astype(int)

data_a = data[(data['enrollment'] >= 20) & (data['enrollment'] <= 60)].copy()

data_a['enrollment_centered'] = data_a['enrollment'] - 41


stage1_a = smf.ols('classize ~ above_41 + enrollment_centered + perc_disadvantaged', data=data_a).fit()
data_a['classize_fitted'] = stage1_a.fittedvalues

model_a = smf.ols('avgmath ~ classize_fitted + enrollment_centered + perc_disadvantaged', data=data_a).fit()

print("Fuzzy RDD with Cutoff 41")
print(model_a.summary().tables[1])

data['predicted_size'] = data['enrollment'] / (np.floor((data['enrollment'] - 1) / 40) + 1)

data['above_threshold'] = ((data['enrollment'] - 1) % 40 >= 0).astype(int)

stage1_b = smf.ols('predicted_size ~ above_threshold + perc_disadvantaged', data=data).fit()
data['predicted_size_fitted'] = stage1_b.fittedvalues

model_b = smf.ols('avgmath ~ predicted_size_fitted + perc_disadvantaged', data=data).fit()

print("\n Fuzzy RDD with all Cutoff")
print(model_b.summary().tables[1])

with open('fuzzy_rdd_results.txt', 'w', encoding='utf-8') as f:
    f.write("Section (a): Fuzzy RDD at threshold 41\n")
    f.write(model_a.summary().as_text())
    f.write("\nSection (b): Fuzzy RDD with all thresholds (similar to column 8, Table IV)\n")
    f.write(model_b.summary().as_text())
```

    Fuzzy RDD with Cutoff 41
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              81.2981      3.985     20.403      0.000      73.475      89.121
    classize_fitted        -0.3776      0.151     -2.505      0.012      -0.673      -0.082
    enrollment_centered     0.0602      0.030      2.000      0.046       0.001       0.119
    perc_disadvantaged     -0.3535      0.020    -17.269      0.000      -0.394      -0.313
    =======================================================================================
    
     Fuzzy RDD with all Cutoff
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                 0.0667      0.000    309.254      0.000       0.066       0.067
    predicted_size_fitted     2.1975      0.008    269.527      0.000       2.182       2.214
    perc_disadvantaged       -0.0569      0.013     -4.388      0.000      -0.082      -0.031
    =========================================================================================
    

# Q_16


```python

data['large_class'] = (data['enrollment'] >= 41).astype(int)
data_a = data[(data['enrollment'] >= 20) & (data['enrollment'] <= 60)].copy()
data_a['enrollment_centered'] = data_a['enrollment'] - 41


model_14a_with = smf.ols('avgmath ~ large_class + enrollment_centered + perc_disadvantaged', data=data_a).fit()

model_14a_without = smf.ols('avgmath ~ large_class + enrollment_centered', data=data_a).fit()



data['predicted_size'] = data['enrollment'] / (np.floor((data['enrollment'] - 1) / 40) + 1)


model_14b_with = smf.ols('avgmath ~ predicted_size + perc_disadvantaged', data=data).fit()

model_14b_without = smf.ols('avgmath ~ predicted_size', data=data).fit()



data_a['above_41'] = (data_a['enrollment'] >= 41).astype(int)


stage1_15a = smf.ols('classize ~ above_41 + enrollment_centered + perc_disadvantaged', data=data_a).fit()
data_a['classize_fitted'] = stage1_15a.fittedvalues


model_15a_with = smf.ols('avgmath ~ classize_fitted + enrollment_centered + perc_disadvantaged', data=data_a).fit()

model_15a_without = smf.ols('avgmath ~ classize_fitted + enrollment_centered', data=data_a).fit()



data['above_threshold'] = ((data['enrollment'] - 1) % 40 >= 0).astype(int)


stage1_15b = smf.ols('predicted_size ~ above_threshold + perc_disadvantaged', data=data).fit()
data['predicted_size_fitted'] = stage1_15b.fittedvalues


model_15b_with = smf.ols('avgmath ~ predicted_size_fitted + perc_disadvantaged', data=data).fit()

model_15b_without = smf.ols('avgmath ~ predicted_size_fitted', data=data).fit()


print("Exercise 14(a): Sharp RDD at threshold 41")
print("with perc_disadvantaged:")
print(model_14a_with.summary().tables[1])
print("without perc_disadvantaged:")
print(model_14a_without.summary().tables[1])

print("\n Exercise 14(b): Sharp RDD with all thresholds")
print("with perc_disadvantaged:")
print(model_14b_with.summary().tables[1])
print("without perc_disadvantaged:")
print(model_14b_without.summary().tables[1])

print("\n Exercise 15(a): Fuzzy RDD at threshold 41")
print("with perc_disadvantaged:")
print(model_15a_with.summary().tables[1])
print("without perc_disadvantaged:")
print(model_15a_without.summary().tables[1])

print("\n Exercise 15(b): Fuzzy RDD with all thresholds")
print("with perc_disadvantaged:")
print(model_15b_with.summary().tables[1])
print("without perc_disadvantaged:")
print(model_15b_without.summary().tables[1])

with open('sensitivity_results.txt', 'w', encoding='utf-8') as f:
    f.write("Exercise 14(a): Sharp RDD at threshold 41\n")
    f.write("With perc_disadvantaged:\n")
    f.write(model_14a_with.summary().as_text())
    f.write("\nWithout perc_disadvantaged:\n")
    f.write(model_14a_without.summary().as_text())
    f.write("\nExercise 14(b): Sharp RDD with all thresholds\n")
    f.write("With perc_disadvantaged:\n")
    f.write(model_14b_with.summary().as_text())
    f.write("\nWithout perc_disadvantaged:\n")
    f.write(model_14b_without.summary().as_text())
    f.write("\nExercise 15(a): Fuzzy RDD at threshold 41\n")
    f.write("With perc_disadvantaged:\n")
    f.write(model_15a_with.summary().as_text())
    f.write("\nWithout perc_disadvantaged:\n")
    f.write(model_15a_without.summary().as_text())
    f.write("\nExercise 15(b): Fuzzy RDD with all thresholds\n")
    f.write("With perc_disadvantaged:\n")
    f.write(model_15b_with.summary().as_text())
    f.write("\nWithout perc_disadvantaged:\n")
    f.write(model_15b_without.summary().as_text())
```

    Exercise 14(a): Sharp RDD at threshold 41
    with perc_disadvantaged:
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              69.4308      0.945     73.493      0.000      67.576      71.286
    large_class             3.4231      1.366      2.505      0.012       0.740       6.106
    enrollment_centered    -0.0857      0.055     -1.557      0.120      -0.194       0.022
    perc_disadvantaged     -0.3395      0.019    -17.415      0.000      -0.378      -0.301
    =======================================================================================
    without perc_disadvantaged:
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              62.9015      1.039     60.568      0.000      60.862      64.941
    large_class             2.5906      1.635      1.584      0.114      -0.620       5.802
    enrollment_centered     0.0007      0.066      0.011      0.991      -0.128       0.130
    =======================================================================================
    
     Exercise 14(b): Sharp RDD with all thresholds
    with perc_disadvantaged:
    ======================================================================================
                             coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------------
    Intercept             72.6827      1.075     67.609      0.000      70.574      74.791
    predicted_size        -0.0126      0.032     -0.396      0.692      -0.075       0.050
    perc_disadvantaged    -0.3543      0.014    -24.621      0.000      -0.383      -0.326
    ======================================================================================
    without perc_disadvantaged:
    ==================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    Intercept         60.5326      1.089     55.596      0.000      58.397      62.668
    predicted_size     0.2186      0.034      6.336      0.000       0.151       0.286
    ==================================================================================
    
     Exercise 15(a): Fuzzy RDD at threshold 41
    with perc_disadvantaged:
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              81.2981      3.985     20.403      0.000      73.475      89.121
    classize_fitted        -0.3776      0.151     -2.505      0.012      -0.673      -0.082
    enrollment_centered     0.0602      0.030      2.000      0.046       0.001       0.119
    perc_disadvantaged     -0.3535      0.020    -17.269      0.000      -0.394      -0.313
    =======================================================================================
    without perc_disadvantaged:
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              53.7122      4.361     12.317      0.000      45.151      62.274
    classize_fitted         0.4225      0.171      2.466      0.014       0.086       0.759
    enrollment_centered     0.0567      0.036      1.577      0.115      -0.014       0.127
    =======================================================================================
    
     Exercise 15(b): Fuzzy RDD with all thresholds
    with perc_disadvantaged:
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept                 0.0667      0.000    309.254      0.000       0.066       0.067
    predicted_size_fitted     2.1975      0.008    269.527      0.000       2.182       2.214
    perc_disadvantaged       -0.0569      0.013     -4.388      0.000      -0.082      -0.031
    =========================================================================================
    without perc_disadvantaged:
    =========================================================================================
                                coef    std err          t      P>|t|      [0.025      0.975]
    -----------------------------------------------------------------------------------------
    Intercept               -13.8301      3.167     -4.367      0.000     -20.041      -7.619
    predicted_size_fitted     2.6205      0.102     25.661      0.000       2.420       2.821
    =========================================================================================
    

# Q_17


```python
plt.figure(figsize=(10, 6))
plt.hist(data['enrollment'], bins=50, edgecolor='black')
plt.axvline(x=41, color='red', linestyle='--', label='Cutoff at 41')
plt.title('Distribution of Enrollment')
plt.xlabel('Enrollment')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig('enrollment_histogram.png')
plt.show()

print("Part (a): Check for bunching around the cutoff (enrollment=41).")
print("If there is unusual piling up just before or after 41, it may suggest manipulation.")


data_b = data[(data['enrollment'] >= 20) & (data['enrollment'] <= 60)].copy()
data_b['enrollment_centered'] = data_b['enrollment'] - 41

bins = np.linspace(-21, 19, 20)  # 2-unit bins
data_b['enrollment_bin'] = pd.cut(data_b['enrollment_centered'], bins, labels=bins[:-1])
bin_means = data_b.groupby('enrollment_bin', observed=True)['avgmath'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.scatter(bin_means['enrollment_bin'], bin_means['avgmath'], color='blue', label='Local Averages')
plt.axvline(x=0, color='red', linestyle='--', label='Cutoff (enrollment=41)')
plt.title('RDD Plot: Math Scores vs. Enrollment')
plt.xlabel('Centered Enrollment (enrollment - 41)')
plt.ylabel('Average Math Score (avgmath)')
plt.legend()
plt.grid(True)
plt.savefig('rdd_binscatter.png')
plt.show()

print("Part (b): Is there a clear discontinuity in avgmath at the cutoff (enrollment=41)?")

# Part (c)
left = data_b[data_b['enrollment_centered'] < 0]
right = data_b[data_b['enrollment_centered'] >= 0]

model_left_linear = smf.ols('avgmath ~ enrollment_centered', data=left).fit()
model_right_linear = smf.ols('avgmath ~ enrollment_centered', data=right).fit()

left['enrollment_centered_sq'] = left['enrollment_centered'] ** 2
right['enrollment_centered_sq'] = right['enrollment_centered'] ** 2
model_left_quad = smf.ols('avgmath ~ enrollment_centered + enrollment_centered_sq', data=left).fit()
model_right_quad = smf.ols('avgmath ~ enrollment_centered + enrollment_centered_sq', data=right).fit()

plt.figure(figsize=(10, 6))
plt.scatter(bin_means['enrollment_bin'], bin_means['avgmath'], color='blue', label='Local Averages')
plt.axvline(x=0, color='red', linestyle='--', label='Cutoff')

x_left = np.linspace(-21, 0, 100)
x_right = np.linspace(0, 19, 100)
plt.plot(x_left, model_left_linear.predict(pd.DataFrame({'enrollment_centered': x_left})), 'g-', label='Linear Trend (Left)')
plt.plot(x_right, model_right_linear.predict(pd.DataFrame({'enrollment_centered': x_right})), 'g--', label='Linear Trend (Right)')

plt.plot(x_left, model_left_quad.predict(pd.DataFrame({'enrollment_centered': x_left, 'enrollment_centered_sq': x_left**2})), 'b-', label='Quadratic Trend (Left)')
plt.plot(x_right, model_right_quad.predict(pd.DataFrame({'enrollment_centered': x_right, 'enrollment_centered_sq': x_right**2})), 'b--', label='Quadratic Trend (Right)')

plt.title('RDD Plot with Linear and Quadratic Trends')
plt.xlabel('Centered Enrollment (enrollment - 41)')
plt.ylabel('Average Math Score (avgmath)')
plt.legend()
plt.grid(True)
plt.savefig('rdd_trends.png')
plt.show()

print("Part (c): Do the linear or quadratic trends capture nonlinearities near the cutoff?")

# Part (d)
bandwidths = [10, 15, 20]
orders = [1, 2]
data['large_class'] = (data['enrollment'] >= 41).astype(int)

print("\nPart (d): Sensitivity to Bandwidth and Polynomial Order")
for bw in bandwidths:
    for order in orders:
        # Restrict data to bandwidth
        data_bw = data[(data['enrollment'] >= 41 - bw) & (data['enrollment'] <= 41 + bw)].copy()
        data_bw['enrollment_centered'] = data_bw['enrollment'] - 41
        if order == 2:
            data_bw['enrollment_centered_sq'] = data_bw['enrollment_centered'] ** 2
            formula = 'avgmath ~ large_class + enrollment_centered + enrollment_centered_sq + perc_disadvantaged'
        else:
            formula = 'avgmath ~ large_class + enrollment_centered + perc_disadvantaged'
        model = smf.ols(formula, data=data_bw).fit()
        print(f"\nBandwidth: {bw}, Polynomial Order: {order}")
        print(model.summary().tables[1])

# Part (e)
data_b['enrollment_centered_sq'] = data_b['enrollment_centered'] ** 2
model_placebo = smf.ols('perc_disadvantaged ~ large_class + enrollment_centered + enrollment_centered_sq', data=data_b).fit()
print("\nPart (e): Placebo RDD with perc_disadvantaged as Outcome")
print(model_placebo.summary().tables[1])
print("Is the coefficient of large_class significant? It should be insignificant, as perc_disadvantaged should not jump at the cutoff.")
```


    
![png](output_15_0.png)
    


    Part (a): Check for bunching around the cutoff (enrollment=41).
    If there is unusual piling up just before or after 41, it may suggest manipulation.
    


    
![png](output_15_2.png)
    


    Part (b): Is there a clear discontinuity in avgmath at the cutoff (enrollment=41)?
    

    C:\Users\SHAYAN\AppData\Local\Temp\ipykernel_7052\1745654502.py:49: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      left['enrollment_centered_sq'] = left['enrollment_centered'] ** 2
    C:\Users\SHAYAN\AppData\Local\Temp\ipykernel_7052\1745654502.py:50: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      right['enrollment_centered_sq'] = right['enrollment_centered'] ** 2
    


    
![png](output_15_5.png)
    


    Part (c): Do the linear or quadratic trends capture nonlinearities near the cutoff?
    
    Part (d): Sensitivity to Bandwidth and Polynomial Order
    
    Bandwidth: 10, Polynomial Order: 1
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              69.2282      1.311     52.821      0.000      66.650      71.806
    large_class             2.8431      1.878      1.514      0.131      -0.850       6.537
    enrollment_centered     0.0729      0.150      0.486      0.627      -0.222       0.368
    perc_disadvantaged     -0.3415      0.028    -12.064      0.000      -0.397      -0.286
    =======================================================================================
    
    Bandwidth: 10, Polynomial Order: 2
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 66.2144      1.741     38.036      0.000      62.790      69.638
    large_class                5.3933      2.105      2.563      0.011       1.254       9.533
    enrollment_centered       -0.1671      0.175     -0.955      0.340      -0.511       0.177
    enrollment_centered_sq     0.0435      0.017      2.603      0.010       0.011       0.076
    perc_disadvantaged        -0.3361      0.028    -11.935      0.000      -0.391      -0.281
    ==========================================================================================
    
    Bandwidth: 15, Polynomial Order: 1
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              68.6643      1.082     63.473      0.000      66.539      70.789
    large_class             4.1153      1.552      2.652      0.008       1.067       7.164
    enrollment_centered    -0.1320      0.084     -1.575      0.116      -0.297       0.033
    perc_disadvantaged     -0.3336      0.023    -14.322      0.000      -0.379      -0.288
    =======================================================================================
    
    Bandwidth: 15, Polynomial Order: 2
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 67.5565      1.383     48.856      0.000      64.840      70.273
    large_class                5.0278      1.706      2.948      0.003       1.677       8.379
    enrollment_centered       -0.1885      0.095     -1.992      0.047      -0.374      -0.003
    enrollment_centered_sq     0.0078      0.006      1.285      0.199      -0.004       0.020
    perc_disadvantaged        -0.3331      0.023    -14.306      0.000      -0.379      -0.287
    ==========================================================================================
    
    Bandwidth: 20, Polynomial Order: 1
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept              69.3459      0.930     74.558      0.000      67.520      71.172
    large_class             3.3664      1.353      2.488      0.013       0.710       6.023
    enrollment_centered    -0.0726      0.055     -1.331      0.184      -0.180       0.035
    perc_disadvantaged     -0.3412      0.020    -17.088      0.000      -0.380      -0.302
    =======================================================================================
    
    Bandwidth: 20, Polynomial Order: 2
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 68.4216      1.162     58.890      0.000      66.140      70.703
    large_class                4.0484      1.447      2.798      0.005       1.208       6.889
    enrollment_centered       -0.1005      0.058     -1.719      0.086      -0.215       0.014
    enrollment_centered_sq     0.0037      0.003      1.326      0.185      -0.002       0.009
    perc_disadvantaged        -0.3401      0.020    -17.030      0.000      -0.379      -0.301
    ==========================================================================================
    
    Part (e): Placebo RDD with perc_disadvantaged as Outcome
    ==========================================================================================
                                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------------
    Intercept                 19.9866      2.045      9.773      0.000      15.971      24.002
    large_class                1.9986      2.747      0.728      0.467      -3.394       7.391
    enrollment_centered       -0.2426      0.108     -2.241      0.025      -0.455      -0.030
    enrollment_centered_sq    -0.0033      0.005     -0.651      0.515      -0.013       0.007
    ==========================================================================================
    Is the coefficient of large_class significant? It should be insignificant, as perc_disadvantaged should not jump at the cutoff.
    


```python

```
