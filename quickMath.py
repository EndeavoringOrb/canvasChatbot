p0 = (70, 2)
p1 = (90, 1)

changeX = p1[0] - p0[0]
changeY = p1[1] - p0[1]

midX = (p0[0] + p1[0]) / 2
midY = (p0[1] + p1[1]) / 2

pctX = changeX / midX
pctY = changeY / midY

elasticity = pctX / pctY
print(f"Elasticity: {elasticity}")
if abs(elasticity) < 1.0:
    print("Inelastic")
elif abs(elasticity) > 1.0:
    print("Elastic")
else:
    print("Unit Elastic")
