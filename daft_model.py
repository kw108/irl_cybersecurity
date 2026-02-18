import daft

# Colors.
p_color = {"ec": "#46a546"}
s_color = {"ec": "#f89406"}

scaling = 1.0

pgm = daft.PGM(aspect=1.5, node_unit=1.00)

# Nodes.
pgm.add_node("attacker", r"$attacker$", 0 * scaling, 0 * scaling, observed=True, scale=1.2)
pgm.add_node("host_1", r"$gateway$", -0.10 * scaling, 2 * scaling, scale=1.2)
pgm.add_node("host_2", r"$historian$", 0.9 * scaling, 0.9 * scaling, scale=1.2)
pgm.add_node("host_3", r"$HMI$", 2.5 * scaling, 0.1 * scaling, scale=1.2)
pgm.add_node("host_4", r"$RTU\ a$", 1.8 * scaling, 1.8 * scaling, plot_params=p_color, scale=1.2)
pgm.add_node("host_5", r"$RTU\ b$", 2.9 * scaling, 1.9 * scaling, plot_params=p_color, scale=1.2)
pgm.add_node("host_6", r"$RTU\ c$", 2 * scaling, 2.9 * scaling, plot_params=p_color, scale=1.2)
pgm.add_node("host_7", r"$SCADA$", 4 * scaling, 3 * scaling, plot_params=p_color, scale=1.2)
pgm.add_node("host_8", r"$RTU\ d$", 3.8 * scaling, 1 * scaling, plot_params=s_color, scale=1.2)

# Edges.
pgm.add_edge("attacker", "host_1", label=r"$p_{1}$", xoffset=-0.1)
pgm.add_edge("attacker", "host_2", label=r"$p_{2}$", xoffset=-0.1)
pgm.add_edge("attacker", "host_3", label=r"$p_{3}$")
pgm.add_edge("host_1", "host_4", label=r"$p_{14}$")
pgm.add_edge("host_2", "host_4", label=r"$p_{24}$", xoffset=-0.1)
pgm.add_edge("host_3", "host_4", label=r"$p_{34}$")
pgm.add_edge("host_3", "host_5", label=r"$p_{35}$")
pgm.add_edge("host_4", "host_6", label=r"$p_{46}$", xoffset=-0.1, yoffset=-0.1)
pgm.add_edge("host_4", "host_7", label=r"$p_{47}$")
pgm.add_edge("host_5", "host_6", label=r"$p_{56}$")
pgm.add_edge("host_5", "host_7", label=r"$p_{57}$")
pgm.add_edge("host_5", "host_8", label=r"$p_{58}$", xoffset=0.1)
pgm.add_edge("host_6", "host_7", label=r"$p_{67}$")
pgm.add_edge("host_7", "host_8", label=r"$p_{78}$", xoffset=-0.1)

# And a plate.
# pgm.add_plate([1.5, 0.2, 2, 3.2], label=r"exposure $i$", shift=-0.1)
# pgm.add_plate([2, 0.5, 1, 1], label=r"pixel $j$", shift=-0.1)

# Render and save.
pgm.render()
pgm.savefig('daft_model.pdf', dpi=600, bbox_inches='tight', pad_inches=0)
