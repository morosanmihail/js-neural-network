//Neural Network

function NeuralNetwork(inputcount, hiddenlayersize, outputcount) {
	this.inputcount = inputcount;
	this.hiddenlayersize = hiddenlayersize;
	this.outputcount = outputcount;
	
	this.InputLayer = [];
	this.HiddenLayer = [];
	this.OutputLayer = [];
	
	this.Error = [];
	
	for(var i=0;i<=inputcount;i++)
	{
		this.InputLayer[i] = new Neuron();
	}
	this.InputLayer[0].SetOutput(1);
	
	for(var i=0;i<hiddenlayersize;i++)
	{
		this.HiddenLayer[i] = new Neuron();
		this.HiddenLayer[i].SetRandomWeights(inputcount+1);
	}
	
	for(var i=0;i<outputcount;i++)
	{
		this.OutputLayer[i] = new Neuron();
		this.OutputLayer[i].SetRandomWeights(hiddenlayersize);
	}
}

NeuralNetwork.prototype.RunDataSet = function(input) {
	//Do checks here
	for(var i=0;i<this.inputcount;i++)
	{
		if(isNaN(input[i]))
		{
			return [];
		}
	}
	
	//Run through network
	for(var i=1;i<=this.inputcount;i++)
	{
		this.InputLayer[i].SetOutput(input[i-1]);
	}
	
	for(var i=0;i<this.hiddenlayersize;i++)
	{
		this.HiddenLayer[i].CalculateOutput(this.InputLayer);
	}
	
	var result = [];
	
	for(var i=0;i<this.outputcount;i++)
	{
		this.OutputLayer[i].CalculateOutput(this.HiddenLayer);
		result[i] = this.OutputLayer[i].GetOutput();
	}
	
	return result;
};

NeuralNetwork.prototype.TrainNetwork = function(input, desired) {
	var res = this.RunDataSet(input);
	
	if(res.length == 0)
	{
		return [];
	}
	
	for(var i=0;i<desired.length;i++)
	{
		this.Error[i] = desired[i] - res[i];
	}
	
	for(var i=0;i<this.outputcount;i++)
	{
		this.OutputLayer[i].CalculateLocalGradientO(this.Error[i]);
	}
	
	for(var i=0;i<this.hiddenlayersize;i++)
	{
		this.HiddenLayer[i].CalculateLocalGradientH(this.OutputLayer, i);
	}
	
	for(var i=0;i<this.outputcount;i++)
	{
		this.OutputLayer[i].UpdateWeigthsFromGradient(this.HiddenLayer);
	}
	
	for(var i=0;i<this.hiddenlayersize;i++)
	{
		this.HiddenLayer[i].UpdateWeigthsFromGradient(this.InputLayer);
	}
	
	return res;
};

//Neuron

function Neuron() {
	this.Lambda = 0.9;
	this.Eta = 0.9;
	this.Alpha = 0.2;

	this.Weights = [];
	this.PrevWeightDelta = [];
	
	this.output = 0;
	this.LocalGradient = 0;
}

Neuron.prototype.SetWeights = function(weights) {
	for(var i=0;i<weights.length;i++)
	{
		this.Weights[i] = weights[i];
		this.PrevWeightDelta[i] = 0;
	}
}

Neuron.prototype.SetRandomWeights = function(count) {
	for(var i=0;i<count;i++)
	{
		var w = Math.random();
		this.Weights[i] = w;
		this.PrevWeightDelta[i] = 0;
	}
};

Neuron.prototype.SetOutput = function(output) {
	this.output = output;
};

Neuron.prototype.GetOutput = function() {
	return this.output;
};

Neuron.prototype.CalculateOutput = function(previousLayer) {
	var result = 0.0;
	
	for(var i=0;i<previousLayer.length;i++)
	{
		result += previousLayer[i].GetOutput() * this.Weights[i];
	}
	
	if(result < -45.0)
		result = 0;
	else if(result > 45.0)
		result = 1;
	else
		result = 1.0 / ( 1.0 + Math.exp(-result * this.Lambda) );
		
	this.output = result;
};

Neuron.prototype.GetLocalGradient = function() {
	return this.LocalGradient;
};

Neuron.prototype.CalculateLocalGradientO = function(error) {
	this.LocalGradient = this.Lambda * this.output * (1-this.output) * error;
};

Neuron.prototype.CalculateLocalGradientH = function(nextLayer, index) {
	this.LocalGradient = this.Lambda * this.output * (1-this.output);
	
	var r = 0.0;
	for(var i=0;i<nextLayer.length;i++)
	{
		r += nextLayer[i].GetLocalGradient() * nextLayer[i].Weights[index];
	}
	
	this.LocalGradient *= r;
};

Neuron.prototype.UpdateWeigthsFromGradient = function(previousLayer) {
	for(var i=0;i<this.Weights.length;i++)
	{
		var delta = this.Eta * this.LocalGradient * previousLayer[i].GetOutput();
		
		this.Weights[i] = this.Weights[i] + delta + this.Alpha * this.PrevWeightDelta[i];
		
		this.PrevWeightDelta[i] = delta;
	}
};

//Random Utilities

NeuralNetwork.prototype.SaveToLocalStorage = function() {
	if(supports_html5_storage)
	{
		localStorage["OutputLayer"] = JSON.stringify(this.OutputLayer);
		localStorage["HiddenLayer"] = JSON.stringify(this.HiddenLayer);
	}
}

NeuralNetwork.prototype.LoadFromLocalStorage = function() {
	if(supports_html5_storage)
	{
		var OL = JSON.parse(localStorage["OutputLayer"]);
		var HL = JSON.parse(localStorage["HiddenLayer"]);
		
		for(var i=0;i<this.outputcount;i++)
		{
			this.OutputLayer[i] = new Neuron();
			this.OutputLayer[i].SetWeights(OL[i].Weights);
		}
		
		for(var i=0;i<this.hiddenlayersize;i++)
		{
			this.HiddenLayer[i] = new Neuron();
			this.HiddenLayer[i].SetWeights(HL[i].Weights);
		}
	}
}

function supports_html5_storage() {
  try {
    return 'localStorage' in window && window['localStorage'] !== null;
  } catch (e) {
    return false;
  }
}


//Web Worker

function runRandomData(i,TCount,visual,targetFunction) {	
	//for(var i=0;i<TCount;i++)
	{
		var input1 = Math.random();
		var input2 = Math.random();
						
		var input = [input1, input2];
						
		var desired = [targetFunction(input1, input2)];
						
		var res = NN.TrainNetwork(input, desired);
		
		$("#input1").val(input1);
		$("#input2").val(input2);
							
		$("#output").html("Training " + i + ": " + desired + " - " + res);
		
		
			if(i<TCount && visual)
			{
				setTimeout(function() { runRandomData(i+1,TCount,visual, targetFunction)},0);
			}
	}
}
