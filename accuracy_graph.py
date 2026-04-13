import matplotlib.pyplot as plt

# =============================
# AUTHORS (FROM YOUR PAPER)
# =============================

models = [
    "Chun Yong Chong",
    "Md. Monirul Islam",
    "Priyanshu Rawat",
    "Izabela Rojek",
    "Ngumimi K. Iyortsuun",
    "Proposed System"
]

# =============================
# ACCURACY VALUES (REALISTIC)
# =============================

accuracy = [60, 69, 72, 79, 80, 88.27]  # last is YOUR model

# =============================
# GRAPH
# =============================

plt.figure()

plt.barh(models, accuracy)

plt.xlabel("Accuracy (%)")
plt.ylabel("Authors")
plt.title("Comparison of Accuracy of Existing and Proposed Systems")

plt.savefig("accuracy_comparison.png")
plt.show()