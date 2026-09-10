import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
file_path = "/Users/pang/Desktop/ProMP_pca_2way_anova/ProMP/vis/digit_library_l2_norm_statistics.csv"   # change path if needed
df = pd.read_csv(file_path)

# Convert Task to string for categorical axis
df["Task"] = df["Task"].astype(str)

# Grouped bar positions
x = np.arange(len(df))
width = 0.35

# A commonly used “Nature-style” 2-color pair (blue + red/orange)
nature_blue = "#4DBBD5"
nature_red  = "#E64B35" 

# nature_blue = "#6EC6DF"   # lighter blue
# nature_red  = "#F07A6A" 

plt.figure(figsize=(8,6))



# Personalized bars + std
plt.bar(
    x + width / 2,
    df["Personalized_Mean"],
    width,
    yerr=df["Personalized_Std"],
    capsize=4,
    label="Personalized",
    color=nature_blue,
)
# Generic bars + std
plt.bar(
    x - width / 2,
    df["Generic_Mean"],
    width,
    yerr=df["Generic_Std"],
    capsize=4,
    label="Generic",
    color=nature_red,
)

ax = plt.gca()

# Compute maximum including error bars
y_max = max(
    (df["Generic_Mean"] + df["Generic_Std"]).max(),
    (df["Personalized_Mean"] + df["Personalized_Std"]).max()
)

# Set y-ticks every 40
ax.set_yticks(np.arange(0, 200, 40))

# Increase y-tick font size
ax.tick_params(axis='y')
plt.xticks(x, df["Task"],fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlabel("Task: Digits writing",fontsize = 16)
plt.ylabel("RMSE",fontsize = 16)
plt.title("Pixel-wise error per Task: Personalized vs Generic", fontsize = 16)
plt.legend()

plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load CSV file
# file_path = "/Users/pang/Desktop/ProMP_pca_2way_anova/ProMP/vis/alphabet_library_iou_statistics.csv"   # change path if needed
# df = pd.read_csv(file_path)

# # Convert Task to string for categorical axis
# df["Task"] = df["Task"].astype(str)

# # Grouped bar positions
# x = np.arange(len(df))
# width = 0.35

# # A commonly used “Nature-style” 2-color pair (blue + red/orange)
# # nature_blue = "#4DBBD5"
# # nature_red  = "#E64B35" 

# nature_blue = "#6EC6DF"   # lighter blue
# nature_red  = "#F07A6A" 

# plt.figure(figsize=(8,6))



# # Personalized bars + std
# plt.bar(
#     x + width / 2,
#     df["Personalized_Mean"],
#     width,
#     yerr=df["Personalized_Std"],
#     capsize=4,
#     label="Personalized",
#     color=nature_blue,
# )
# # Generic bars + std
# plt.bar(
#     x - width / 2,
#     df["Generic_Mean"],
#     width,
#     yerr=df["Generic_Std"],
#     capsize=4,
#     label="Generic",
#     color=nature_red,
# )

# ax = plt.gca()

# # Compute maximum including error bars
# y_max = max(
#     (df["Generic_Mean"] + df["Generic_Std"]).max(),
#     (df["Personalized_Mean"] + df["Personalized_Std"]).max()
# )

# # Set y-ticks every 40
# ax.set_yticks(np.arange(0, 1.2, 0.2))

# # Increase y-tick font size
# ax.tick_params(axis='y')
# plt.xticks(x, df["Task"],fontsize = 16)
# plt.yticks(fontsize = 16)
# plt.xlabel("Task: Lower Case Letter Writing",fontsize = 16)
# plt.ylabel("IoU",fontsize = 16)
# plt.title("IoU per Task: Personalized vs Generic", fontsize = 16)
# plt.legend()

# plt.tight_layout()
# plt.show()