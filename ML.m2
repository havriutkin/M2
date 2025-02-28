
-- -*- coding: utf-8 -*-
newPackage(
    "ML",
    Version => "0,1",
    Date => "February 13, 2025",
    Authors => {
	{Name => "Vladyslav Havriutkin", 
	    Email => "havriutkin@gmail.com", 
	    HomePage => "http://www.math.uiuc.edu/~doe/"}},
    Headline => "Interface for Python  ML procedures",
    Keywords => {"Documentation"},
    DebuggingMode => false
    );

needsPackage "Python"
torch = import "torch"
np = import "numpy"

loadModel = (modelPath) -> (
    model = torch@@load(toString modelPath);
    model@@eval();
    return model;
);

runModel = (modelPath, inputList) -> (
    command = "python3 forward_model.py --model_path " | modelPath | " --input " | inputList;
    result = run command;
    return result;
);

end--
restart;

needsPackage "ML"

myPath = "simple_sum_model.pt";
myInput = "1,2,3,4";
myRes = runModel(myPath, myInput);
print myRes;

