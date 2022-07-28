// RUN: soda-opt -pass-pipeline="func.func(test-data-flow-graph {loop=1})" %s
// RUN: FileCheck %s -input-file=dfg.graphml --check-prefixes=CHECK_FILE

func.func @example(%arg0: memref<10xi32>, %arg1: memref<10xi32>) {
	affine.for %arg2 = 0 to 10 {
		%0 = affine.load %arg0[%arg2] : memref<10xi32>
		%1 = arith.muli %0, %0 : i32
		affine.store %1, %arg1[%arg2] : memref<10xi32>
	}
	return
}

// CHECK_FILE: <?xml version="1.0" encoding="UTF-8"?>
// CHECK_FILE: <graphml>
// CHECK_FILE:     <graph>
// CHECK_FILE:     <node id="1">
// CHECK_FILE:         <data key="name">affine.load_1</data>
// CHECK_FILE:         <data key="uses_resource">mem</data>
// CHECK_FILE:     </node>
// CHECK_FILE:     <node id="2">
// CHECK_FILE:         <data key="name">arith.muli_2</data>
// CHECK_FILE:         <data key="uses_resource">arith.muli</data>
// CHECK_FILE:     </node>
// CHECK_FILE:     <node id="3">
// CHECK_FILE:         <data key="name">affine.store_3</data>
// CHECK_FILE:         <data key="uses_resource">mem</data>
// CHECK_FILE:     </node>
// CHECK_FILE:     <edge id="1_2" source="1" target="2">
// CHECK_FILE:         <data key="delay">0</data>
// CHECK_FILE:         <data key="distance">0</data>
// CHECK_FILE:         <data key="deptype">Precedence</data>
// CHECK_FILE:     </edge>
// CHECK_FILE:     <edge id="1_2" source="1" target="2">
// CHECK_FILE:         <data key="delay">0</data>
// CHECK_FILE:         <data key="distance">0</data>
// CHECK_FILE:         <data key="deptype">Precedence</data>
// CHECK_FILE:     </edge>
// CHECK_FILE:     <edge id="2_3" source="2" target="3">
// CHECK_FILE:         <data key="delay">0</data>
// CHECK_FILE:         <data key="distance">0</data>
// CHECK_FILE:         <data key="deptype">Precedence</data>
// CHECK_FILE:     </edge>
// CHECK_FILE:     </graph>
// CHECK_FILE: </graphml>
