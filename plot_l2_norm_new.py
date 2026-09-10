# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load CSV file
# file_path = "/Users/pang/Desktop/ProMP_pca_2way_anova/ProMP/vis_vae_style_alphabet_user1/alphabet_library_l2_norm_statistics_new_new.csv"
# df = pd.read_csv(file_path)

# # Convert Task to string for categorical axis
# df["Task"] = df["Task"].astype(str)

# # Grouped bar positions
# x = np.arange(len(df))
# width = 0.25

# # Nature / Okabe-Ito palette
# nature_blue = "#0072B2"
# nature_orange = "#E69F00"
# nature_red = "#D55E00"

# plt.figure(figsize=(8,6))

# # Generic bars
# plt.bar(
#     x - width,
#     df["Generic_Mean"],
#     width,
#     yerr=df["Generic_Std"],
#     capsize=4,
#     label="Generic",
#     color=nature_red,
# )

# # Personalized PCA bars
# plt.bar(
#     x,
#     df["Personalized_Mean_PCA"],
#     width,
#     yerr=df["Personalized_Std_PCA"],
#     capsize=4,
#     label="Personalized (PCA)",
#     color=nature_blue,
# )

# # Personalized VAE bars
# plt.bar(
#     x + width,
#     df["Personalized_Mean_vae"],
#     width,
#     yerr=df["Personalized_Std_vae"],
#     capsize=4,
#     label="Personalized (VAE)",
#     color=nature_orange,
# )

# ax = plt.gca()

# # Set y-ticks every 40
# ax.set_yticks(np.arange(0, 200, 40))

# # Tick formatting
# plt.xticks(x, df["Task"], fontsize=16)
# plt.yticks(fontsize=16)

# plt.xlabel("Task: Digits writing", fontsize=16)
# plt.ylabel("RMSE", fontsize=16)
# plt.title("Pixel-wise error per Task", fontsize=16)

# plt.legend()

# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV file
file_path = "/Users/pang/Desktop/ProMP_pca_2way_anova/ProMP/vis_vae_style_alphabet_user1/alphabet_library_iou_statistics_new_new.csv"
df = pd.read_csv(file_path)

# Convert Task to string for categorical axis
df["Task"] = df["Task"].astype(str)

# Grouped bar positions
x = np.arange(len(df))
width = 0.25

# Nature / Okabe-Ito palette
nature_blue = "#0072B2"
nature_orange = "#E69F00"
nature_red = "#D55E00"

plt.figure(figsize=(8,6))

# Generic bars
plt.bar(
    x - width,
    df["Generic_Mean"],
    width,
    yerr=df["Generic_Std"],
    capsize=4,
    label="Generic",
    color=nature_red,
)

# Personalized PCA bars
plt.bar(
    x,
    df["Personalized_Mean_PCA"],
    width,
    yerr=df["Personalized_Std_PCA"],
    capsize=4,
    label="Personalized (PCA)",
    color=nature_blue,
)

# Personalized VAE bars
plt.bar(
    x + width,
    df["Personalized_Mean_vae"],
    width,
    yerr=df["Personalized_Std_vae"],
    capsize=4,
    label="Personalized (VAE)",
    color=nature_orange,
)

ax = plt.gca()

# Set y-ticks every 40
ax.set_yticks(np.arange(0, 1.2, 0.2))

# Tick formatting
plt.xticks(x, df["Task"], fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel("Task: Digits writing", fontsize=16)
plt.ylabel("IoU", fontsize=16)
plt.title("IoU per Task", fontsize=16)

plt.legend()

plt.tight_layout()
plt.show()