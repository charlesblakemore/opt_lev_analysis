////////////////////////////////////////////////// 
// gmsh geometry specification for cube
// homer reid
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// geometric parameters 
//////////////////////////////////////////////////
aspect = 10;
depth = 10;
gRatio = 10;
aRatio = 10;
L = aspect*10;   // side length
D = depth*10; //depth
H = 10;
offMax = 45;

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
lCoarse = grid*gRatio;
lFinel  =  grid*((aRatio-1)*((L-H)+2*offMax)/(L-H)+1);
lFiner  =  grid*((aRatio-1)*((L-H)-2*offMax)/(L-H)+1);

//////////////////////////////////////////////////
// geometric description of cube /////////////////
//////////////////////////////////////////////////
Point(1) = { L/2, -H/2, -D, lCoarse};
Point(2) = {-L/2, -H/2, -D, lCoarse};
Point(3) = {-L/2, -H/2,  0, lFinel};
Point(4) = { L/2, -H/2,  0, lFiner};

Point(5) = { L/2, H/2, -D, lCoarse};
Point(6) = {-L/2, H/2, -D, lCoarse};
Point(7) = {-L/2, H/2,  0, lFinel};
Point(8) = { L/2, H/2,  0, lFiner};

Point(9) = { (L/2)-H, -H/2,  0, grid};
Point(10) = { (L/2)-H, H/2,  0, grid};

Point(13) = { (L/2)-H, -H/2,  -D, lCoarse};
Point(14) = { (L/2)-H, H/2,  -D, lCoarse};

Line(4) = {9, 10};
Line(5) = {10, 14};

Line(11) = {5, 8};
Line(14) = {4, 1};
Line(15) = {1, 5};
Line(16) = {8, 4};
Line(18) = {14, 13};

Line(20) = {13, 9};
Line(21) = {3, 9};
Line(22) = {3, 7};
Line(23) = {7, 10};
Line(24) = {7, 6};
Line(25) = {6, 14};
Line(26) = {2, 3};
Line(27) = {2, 6};
Line(28) = {13, 2};
Line(29) = {8, 10};
Line(30) = {9, 4};
Line(31) = {1, 13};
Line(32) = {14, 5};
Line Loop(33) = {4, -29, 16, -30};
Ruled Surface(34) = {33};
Line Loop(35) = {16, 14, 15, 11};
Ruled Surface(36) = {35};
Line Loop(37) = {11, 29, 5, 32};
Ruled Surface(38) = {37};
Line Loop(39) = {30, 14, 31, 20};
Ruled Surface(40) = {39};
Line Loop(41) = {15, -32, 18, -31};
Ruled Surface(42) = {41};
Line Loop(43) = {28, 27, 25, 18};
Ruled Surface(44) = {43};
Line Loop(45) = {25, -5, -23, 24};
Ruled Surface(46) = {45};
Line Loop(47) = {21, 4, -23, -22};
Ruled Surface(48) = {47};
Line Loop(49) = {24, -27, 26, 22};
Ruled Surface(50) = {49};
Line Loop(51) = {26, 21, -20, 28};
Ruled Surface(52) = {51};
