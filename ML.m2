
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
    	model = torch@@load(toString modelPath)
	model@@eval()
	return model
    );

runModel = (model, inputList) -> (
    	inputArray = np@@array(toPython inputList)
	inputTensor = torch@@tensor(inputArray, torch@@float)
	getattr(torch, "set_grad_enabled")(toPython false)
	output = model(inputTensor) -- todo: figure out how this call works in python interface
	getattr(torch, "set_grad_enabled")(toPython true)
	
	return output
    );



end--
restart;
