
# Generating a basic draw.io template for the diagram based on its elements
drawio_template = """
<mxGraphModel dx="1138" dy="715" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <!-- Nodes -->
    <mxCell id="client1" value="Client 1.0" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="50" y="50" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="site1" value="Site 1.0" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="50" y="150" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="efw1" value="FW 1.0" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="200" y="100" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="fw2" value="FW 2.0" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="400" y="150" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="dc10" value="DC 1.0" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="150" y="300" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="dc20prod" value="DC 2.0 Prod" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="350" y="300" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="dc20legacy" value="DC 2.0 Legacy" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="500" y="300" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="vrf1" value="VRF 1" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="100" y="400" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="vrf2" value="VRF 2" style="ellipse;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="300" y="400" width="80" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="aci" value="ACI Contract" style="rectangle;whiteSpace=wrap;html=1;" vertex="1" parent="1">
      <mxGeometry x="300" y="200" width="120" height="40" as="geometry"/>
    </mxCell>
    <!-- Edges -->
    <mxCell id="edge1" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="client1" target="efw1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="edge2" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="efw1" target="fw2">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="edge3" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="fw2" target="aci">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="edge4" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="dc10" target="dc20prod">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="edge5" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="dc20prod" target="dc20legacy">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="edge6" value="" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.5;entryY=1;entryDx=0;entryDy=0;" edge="1" parent="1" source="vrf1" target="vrf2">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
  </root>
</mxGraphModel>
"""

# Save to a .drawio file for user to download and import into draw.io
drawio_file_path = "/home/ubuntu/projects/dc_migration_diagram.drawio"

with open(drawio_file_path, "w") as f:
    f.write(drawio_template)

drawio_file_path
