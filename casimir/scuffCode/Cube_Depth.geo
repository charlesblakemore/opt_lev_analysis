////////////////////////////////////////////////// 
// gmsh geometry specification for cube
// homer reid
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// geometric parameters 
//////////////////////////////////////////////////
aspect = 10;
depth = 1;
gRatio = 30;
L = aspect*10;   // side length
D = depth*10; //depth
H = 10;

//////////////////////////////////////////////////
// this factor may be increased or decreased to   
// make the meshing more coarse or more fine in a
// uniform way over the entire object 
//////////////////////////////////////////////////
Mesh.CharacteristicLengthFactor=1.3;

//////////////////////////////////////////////////
// these factors may be configured separately
// to make the meshing more coarse or more fine in
// particular regions of the object 
//////////////////////////////////////////////////
grid = DefineNumber[ 0.3, Name "Parameters/grid" ];
lCoarse =  grid*gRatio;
lFine   =  grid*aspect;

//////////////////////////////////////////////////
// geometric description of cube /////////////////
//////////////////////////////////////////////////
Point(1) = { L/2, -H/2, -D, lCoarse};
Point(2) = {-L/2, -H/2, -D, lCoarse};
Point(3) = {-L/2, -H/2,  0, lFine};
Point(4) = { L/2, -H/2,  0, lFine};

Point(5) = { L/2, H/2, -D, lCoarse};
Point(6) = {-L/2, H/2, -D, lCoarse};
Point(7) = {-L/2, H/2,  0, lFine};
Point(8) = { L/2, H/2,  0, lFine};

Point(9) = { -H/2, -H/2,  0, grid};
Point(10) = { -H/2, H/2,  0, grid};
Point(11) = { H/2, -H/2,  0, grid};
Point(12) = { H/2, H/2,  0, grid};

Point(13) = { -H/2, -H/2,  -D, lCoarse};
Point(14) = { -H/2, H/2,  -D, lCoarse};
Point(15) = { H/2, -H/2,  -D, lCoarse};
Point(16) = { H/2, H/2,  -D, lCoarse};
Line(1) = {10, 12};
Line(2) = {12, 11};
Line(3) = {11, 9};
Line(4) = {9, 10};
Line(5) = {10, 14};
Line(6) = {14, 16};
Line(7) = {16, 12};
Line(8) = {11, 15};
Line(9) = {15, 16};
Line(10) = {16, 5};
Line(11) = {5, 8};
Line(12) = {8, 12};
Line(13) = {11, 4};
Line(14) = {4, 1};
Line(15) = {1, 5};
Line(16) = {8, 4};
Line(17) = {15, 1};
Line(18) = {14, 13};
Line(19) = {13, 15};
Line(20) = {13, 9};
Line(21) = {3, 9};
Line(22) = {3, 7};
Line(23) = {7, 10};
Line(24) = {7, 6};
Line(25) = {6, 14};
Line(26) = {2, 3};
Line(27) = {2, 6};
Line(28) = {13, 2};
Line Loop(29) = {4, 1, 2, 3};
Plane Surface(30) = {29};
Line Loop(31) = {2, 13, -16, 12};
Plane Surface(32) = {31};
Line Loop(33) = {23, -4, -21, 22};
Plane Surface(34) = {33};
Line Loop(35) = {1, -7, -6, -5};
Ruled Surface(36) = {35};
Line Loop(37) = {7, -12, -11, -10};
Ruled Surface(38) = {37};
Line Loop(39) = {23, 5, -25, -24};
Ruled Surface(40) = {39};
Line Loop(41) = {10, -15, -17, 9};
Ruled Surface(42) = {41};
Line Loop(43) = {9, -6, 18, 19};
Ruled Surface(44) = {43};
Line Loop(45) = {20, -21, -26, -28};
Ruled Surface(46) = {45};
Line Loop(47) = {26, 22, 24, -27};
Ruled Surface(48) = {47};
Line Loop(49) = {18, 28, 27, 25};
Ruled Surface(50) = {49};
Line Loop(51) = {15, 11, 16, 14};
Ruled Surface(52) = {51};
Line Loop(53) = {14, -17, -8, 13};
Ruled Surface(54) = {53};
Line Loop(55) = {3, -20, 19, -8};
Ruled Surface(56) = {55};
