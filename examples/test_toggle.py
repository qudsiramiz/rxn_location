import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="Line 1"))
fig.add_trace(go.Scatter(x=[2, 3], y=[1, 2], name="Line 2"))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=[
                dict(
                    args=[{"visible": [False]}, {"updatemenus[0].buttons[0].label": "☐"}, [1]],
                    args2=[{"visible": [True]}, {"updatemenus[0].buttons[0].label": "☑"}, [1]],
                    label="☑",
                    method="update"
                )
            ],
            x=1, y=1
        )
    ]
)

fig.write_html("test_toggle.html")
print("Done")
