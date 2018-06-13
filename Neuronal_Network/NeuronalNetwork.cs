using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using BIF4_MLE_NeuronalNetwork.utils;

namespace Neuronal_Network
{
    internal static class NeuronalNetwork
    {
        private static InputLayer InLayer { get; set; } = new InputLayer();
        private static HiddenLayer HidLayer { get; set; } = InLayer.ChildLayer;
        private static OutputLayer OutLayer { get; set; } = HidLayer.ChildLayer;

        private static HashSet<double> NetworkErrorRating { get; set; } = new HashSet<double>();
        private static readonly List<Image> TrainData = new List<Image>();
        private static readonly List<Image> TestData = new List<Image>();
        private static double TargetError { get; set; } = 0.005;

        private static int[,] ConfusionMatrix { get; set; } = new int[10, 10];

        public static double LearningRate { get; private set; } = 0.2;
        public static double MomentumFactor { get; private set; } = 0.9;
        public static bool UseMomentum { get; private set; } = true;

        public static readonly Thread WriteInputWeightsFileThread = new Thread(WriteInputLayerWeightsToFile);
        public static readonly Thread WriteHiddenWeightsFileThread = new Thread(WriteHiddenLayerWeightsToFile);
        private static readonly Stopwatch Watch = new Stopwatch();

        private const string PathInputLayer = "mnist/trained_InputWeights";
        private const string PathHiddenWeights = "mnist/trained_HiddenWeights";



        public static void ReadDataFromFile()
        {
            Console.WriteLine("Reading training data ... ");
            foreach (var data in MnistReader.ReadTrainingData())
            {
                TrainData.Add(data);
            }

            Console.WriteLine("Reading test data ... ");
            foreach (var data in MnistReader.ReadTestData())
            {
                TestData.Add(data);
            }
        }

        public static void Train()
        {
            if (CheckSavedWeights())
            {
                Console.WriteLine("Network have already learnt! Reading Train-Data from file...");
                SetInputWeightsFromFile();
                SetHiddenWeightsFromFile();
                Console.WriteLine("Finished!");
                return;
            }
            Console.WriteLine("Starts to train the neural network ...");
            Watch.Start();
            var actualerror = 1.0;
            var count = 0;
            while (actualerror > TargetError)
            {
                NetworkErrorRating.Clear();//Nach jeden vollständigen durchlauf muss die ErrorListe resetter werden um den neuen Wert nicht zu verfälschen
                foreach (var image in TrainData)
                {
                    count++;
                    SetNeuronValueFromImage(image.Data); //Image data wird in InputLayer eingelesen 
                    SetDesiredValueFromLabel(image
                        .Label); //LabelDaten werden im OutputLayer gespeichert für späteren Vergleich
                    FeedForward();
                    NetworkErrorRating.Add(CalculateError());
                    BackProgagate();
                }
                actualerror = CalculateNetworkErrorAverage();
                TimeSpan ts = Watch.Elapsed;
                Console.WriteLine(
                    $"Error-Rate: {actualerror} Image-Count: {count} in {ts:mm\\:ss\\.ff}.");
            }
            WriteWeightsToFile();
            Console.WriteLine($"Training-Session is finished with Error-Rate: {actualerror}.");
            Watch.Stop();
        }

        public static void Test()
        {
            Console.WriteLine("Starts to test the neural network ...");
            InitializeConfusionMatrix();
            var gesamtElemente = 0;
            var korrekteElemente = 0;
            foreach (var image in TestData)
            {
                gesamtElemente++;
                SetNeuronValueFromImage(image.Data); //Image data wird in InputLayer eingelesen 
                SetDesiredValueFromLabel(image.Label); //LabelDaten werden im OutputLayer gespeichert für späteren Vergleich
                FeedForward();
                for (var i = 0; i < OutLayer.DesiredValues.Length; i++)
                {
                    if (OutLayer.DesiredValues[i].CompareTo(1) != 0) continue;
                    if (OutLayer.GetHighestNeuronIndex() == i) korrekteElemente++;
                }
                ConfusionMatrix[(int)image.Label, OutLayer.GetHighestNeuronIndex()]++;
            }
            Console.WriteLine("Creating Confusion-Matrix ...");
            var accuracy = ((double)korrekteElemente * (double)100 / (double)gesamtElemente);
            PrintConfusionMatrix();
            Console.WriteLine($"Accuracy: {accuracy} %");
        }

        private static void FeedForward()
        {
            InLayer.CalculateNeuronValues();
            HidLayer.CalculateNeuronValues();
            OutLayer.CalculateNeuronValues();
        }

        private static void BackProgagate()
        {
            OutLayer.CalculateErrors();
            HidLayer.CalculateErrors();
            HidLayer.AdjustWeights();
            InLayer.AdjustWeights();
        }

        /// <summary>
        /// Startet die Threads welche die erlernten Gewichte in Files ins Dateisystem schreiben
        /// </summary>
        private static void WriteWeightsToFile()
        {
            WriteHiddenWeightsFileThread.Start();
            WriteInputWeightsFileThread.Start();
        }

        /// <summary>
        /// Speichern der erlernten Gewichte für späteren schnellen Start.
        /// </summary>
        private static void WriteInputLayerWeightsToFile()
        {
            var count = 0;
            Console.WriteLine("Creating File on another Thread ...");
            var sb = new StringBuilder();
            foreach (var weight in InLayer.Weight)
            {
                count++;
                sb.Append(weight + "|");
                if(count % 100000 == 0) Console.WriteLine($"{count} weights are saved...");
            }
            File.WriteAllText(PathInputLayer, sb.ToString());
            Console.WriteLine("Saved succesfully!");
        }

        /// <summary>
        /// Speichern der erlernten Gewichte für späteren schnellen Start.
        /// </summary>
        private static void WriteHiddenLayerWeightsToFile()
        {
            var count = 0;
            Console.WriteLine("Creating File on another Thread ...");
            var sb = new StringBuilder();
            foreach (var weight in HidLayer.Weight)
            {
                count++;
                sb.Append(weight + "|");
                if (count % 100000 == 0) Console.WriteLine($"{count} weights are saved...");
            }
            File.WriteAllText(PathHiddenWeights, sb.ToString());
            Console.WriteLine("Saved succesfully!");
        }

        private static void SetHiddenWeightsFromFile()
        {
            var rawData = File.ReadAllText(PathHiddenWeights);
            var data = rawData.Split('|');
            for (var i = 0; i < data.Length - 1; i++)
            {
                HidLayer.Weight[i] = Convert.ToDouble(data[i]);
            }
        }

        private static void SetInputWeightsFromFile()
        {
            var rawData = File.ReadAllText(PathInputLayer);
            var data = rawData.Split('|');
            for (var i = 0; i < data.Length - 1; i++)
            {
                InLayer.Weight[i] = Convert.ToDouble(data[i]);
            }
        }

        /// <summary>
        /// Überprüfe ob die Gewichte vom InputLayer und HiddenLayer gespeichert wurden, dann kann das lernen übersprungen werden.
        /// </summary>
        /// <returns>true = gelernte Gewichte sind im File vorhanden, False = Gewichte sind nicht vorhanden</returns>
        private static bool CheckSavedWeights()
        {
            return File.Exists(PathInputLayer) && File.Exists(PathHiddenWeights);
        }

        /// <summary>
        /// Berechne die Fehlerdurschnitt vom ganzen Durchlauf der 60 000 gelernten Bildern
        /// </summary>
        /// <returns></returns>
        private static double CalculateNetworkErrorAverage()
        {
            double average = 0.0;
            int count = 0;
            foreach (var error in NetworkErrorRating)
            {
                count++;
                average += error;
            }
            average = average / count;
            return average;
        }

        /// <summary>
        /// Berechne Error Rate für eine Iteration vom Neuronalen Netzwerk.
        /// </summary>
        /// <returns></returns>
        private static double CalculateError()
        {
            double error = 0.0;
            for (var i = 0; i < OutLayer.NeuronValue.Length; i++)
            {
                error += Math.Pow(OutLayer.NeuronValue[i] - OutLayer.DesiredValues[i], 2);
            }
            error = error / OutLayer.NeuronValue.Length;
            return error;
        }

        /// <summary>
        /// Setze den korrekten Wert in das Array welcher im Label gespeichert ist. Markiere diesen mit einer 1 im Array
        /// </summary>
        /// <param name="imageLabel"></param>
        private static void SetDesiredValueFromLabel(double imageLabel)
        {
            ResetDesiredValue();
            for (var i = 0; i < OutLayer.DesiredValuesBlueprint.Length; i++)
            {
                if (OutLayer.DesiredValuesBlueprint[i].CompareTo(imageLabel) != 0) continue; //Wenn der Wert nicht mit den Wert vom Label übereinstimmt, dann überspringe und setz die schleife fort
                OutLayer.DesiredValues[i] = 1.0; //Die Stelle welche der Zahl entspricht setze auf 1 -> alle anderen bleiben auf 0
                break;
            }
        }

        /// <summary>
        /// Resette das Array um wieder das neuen Wert vom Label setzen zu können
        /// </summary>
        private static void ResetDesiredValue()
        {
            for (var i = 0; i < OutLayer.DesiredValues.Length; i++)
            {
                OutLayer.DesiredValues[i] = 0.0; //Die Stelle welche der Zahl entspricht setze auf 1 -> alle anderen bleiben auf 0
            }
        }

        /// <summary>
        /// Wandle die Daten vom Reader um und speicher die Werte in das NeuronValue array. Falls eine Schattierung vom Pixel erkant wird, speicher eine 1 sonst 0
        /// </summary>
        /// <param name="data"></param>
        private static void SetNeuronValueFromImage(double[,] data)
        {
            for (var i = 0; i < data.GetLength(0); i++)
            {
                for (var j = 0; j < data.GetLength(1); j++)
                {
                    InLayer.NeuronValue[i * 28 + j] = (data[i, j] > 1) ? 1 : 0; //Splitte zweidimensionales Array in eindimensionales auf und speicher daten im InputLayer
                }
            }
        }

        private static void InitializeConfusionMatrix()
        {
            for (var i = 0; i < 10; i++)
            {
                for (var j = 0; j < 10; j++)
                {
                    ConfusionMatrix[i, j] = 0; //Alle Werte auf 0 setzen
                }
            }
        }

        private static void PrintConfusionMatrix()
        {
            Console.WriteLine("  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |");
            Console.WriteLine("----------------------------------------------------------------");
            for (var i = 0; i < 10; i++)
            {
                Console.Write(i.ToString() + " |");
                for (var j = 0; j < 10; j++)
                {

                    Console.Write(ConfusionMatrix[i, j].ToString().PadLeft(5, ' ') + "|");

                }
                Console.WriteLine();
                Console.WriteLine("----------------------------------------------------------------");
            }
        }
    }
}
