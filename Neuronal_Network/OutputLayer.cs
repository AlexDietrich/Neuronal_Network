﻿using System;

namespace Neuronal_Network
{
    internal class OutputLayer
    {
        /// <summary>
        /// Initialize all needed variables for the neural network. 28x28 Pixels/Picture = 784
        /// </summary>
        private const int NumberOfNeurons = 10;

        private const int NumberOfParentNeurons = 89;


        public double[] NeuronValue { get; set; } = new double[NumberOfNeurons];
        public double[] Error { get; set; } = new double[NumberOfNeurons];
        public double[] DesiredValues { get; set; } = new double[]
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        public double[] DesiredValuesBlueprint { get; set; } = new double[]
            {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};


        private bool LinearOutput = false; //Kein Linearer Output erwünscht :D 


        private OutputLayer ChildLayer { get; set; } = null;
        private HiddenLayer ParentLayer { get; set; }

        public OutputLayer(HiddenLayer parent)
        {
            ParentLayer = parent;
        }

        public void CalculateNeuronValues()
        {
            double x = 0.0;
            if (ParentLayer == null) return;
            for (var j = 0; j < NumberOfNeurons; j++)
            {
                x = 0.0;
                for (var i = 0; i < NumberOfParentNeurons; i++)
                {
                    x += ParentLayer.NeuronValue[i] * ParentLayer.Weight[i*NumberOfParentNeurons+j];
                }
                x += ParentLayer.Bias[j] * ParentLayer.BiasWeight[j];
                if ((ChildLayer == null) && LinearOutput)
                {
                    NeuronValue[j] = x;
                }
                else
                {
                    double k = Math.Exp(x);
                    NeuronValue[j] = k / (1.0 + k);
                }
            }
        }

        public void CalculateErrors()
        {
            for (var i = 0; i < NumberOfNeurons; i++)
            {
                Error[i] = (DesiredValues[i] - NeuronValue[i]) * NeuronValue[i] * (1.0 - NeuronValue[i]);
            }

        }

        /// <summary>
        /// Auf welchen Index der errechneten Neuronen befindet sich der höchste Wert. Das heißt wofür entscheidet das neuronale netz.
        /// </summary>
        /// <returns></returns>
        public int GetHighestNeuronIndex()
        {
            int actualIndex = 0;
            double actualHighestValue = 0.0;
            for (var i = 0; i < NeuronValue.Length; i++)
            {
                if (NeuronValue[i] < actualHighestValue) continue;
                actualHighestValue = NeuronValue[i];
                actualIndex = i;
            }
            return actualIndex;
        }
    }
}
