
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

export {"loadModel", "runModel", "runModelLocal", "createFeedForwardModel"}

needsPackage "Python"
torch = import "torch"
np = import "numpy"

loadModel = (modelPath) -> (
    -- grab the jit submodule
    jit       := torch@@("jit");
    
    -- load the TorchScript archive
    model      := (jit@@load)(toString modelPath);
    model@@eval();                     -- ensure eval mode
    return model;
);

runModel = (modelPath, inputList) -> (
    command := "python3 forward_model.py --model_path " | modelPath | " --input " | inputList;
    result := run command;
    return result;
);

runModelLocal = (model, inputList) -> (
    -- npArr    := np@@array({inputList});
    npArr    := (getattr(np, "array"))({inputList});
    myTensor   := (getattr(torch, "tensor"))(npArr);
    model@@eval();
    --outTensor := model@@forward(myTensor);
    outTensor := (getattr(model, "forward"))(myTensor);
    
    
    -- convert back to a plain Python list, then to an M2 list
    -- pyList   := outTensor@@tolist();
    pyList   := (getattr(outTensor, "tolist"))();
    return toSequence pyList;
);

createFeedForwardModel = (layerSizes) -> (
    nn          := getattr(torch, "nn");
    Sequential  := getattr(nn, "Sequential");
    Linear      := getattr(nn, "Linear");
    ReLU        := getattr(nn, "ReLU");

    pyLayers := {};                   
    n        := #layerSizes;           -- number of layers

    for i from 0 to n-2 do (
        -- add a Linear from layerSizes[i] → layerSizes[i+1]
        pyLayers = pyLayers | { Linear(layerSizes#i, layerSizes#(i+1)) };
	
        -- after every hidden layer (but not after the final), add ReLU
        if i < n-2 then (
            pyLayers = pyLayers | { ReLU() };
        );
    );
    
    -- Create python sequential model
    model := Sequential();  
    for i from 0 to (#pyLayers - 1) do (
        name := toString i;                 
        -- model@@add_module(name, pyLayers#i);
	(getattr(model, "add_module"))(name, pyLayers#i);
    );
    
    return model;
);


end--
restart;

needsPackage "ML"

myPath = "simple_sum_model.pt";
myInput = {1,2,3,4};
myModel = loadModel(myPath);
myRes = runModelLocal(myModel, myInput);
print myRes;
myModel = createFeedForwardModel({4,10,5,1});
myModel
