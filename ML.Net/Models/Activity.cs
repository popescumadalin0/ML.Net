using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.ML.Data;

namespace ML.Net.Models;

public class ActivityData
{
    [ColumnName("Type")] public string Type { get; set; }
    
    [ColumnName("Name")] public string Name { get; set; }
    
    [ColumnName("Description")] public string Description { get; set; }
    
    [ColumnName("Parameters")] public string Parameters { get; set; }
    
    [ColumnName("Outcome")] public string Outcome { get; set; }
}

public class ActivityPrediction
{
    [ColumnName("PredictedLabel")] public string PredictedOutcome { get; set; }
    
    public float Probability { get; set; }
    public float Score { get; set; }
}