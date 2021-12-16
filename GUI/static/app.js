function onClickedPredictPrice(){
    console.log("Predict price button clicked");
    var distance = document.getElementById('ui_distance');
    var surge_multiplier = document.getElementById('ui_surge_multiplier');
    var cab_name = document.getElementById('ui_cab_name');
    var predictPrice = document.getElementById('predictedPrice');
    var url = "https://fp1-model-price-prediction.herokuapp.com/predict_price";

    $.post(url,{
        distance:parseFloat(distance.value),
        surge_multiplier:parseFloat(surge_multiplier.value),
        cab_name: cab_name.value
    }, function(data, status){
        predictPrice.innerHTML = ''
        predictPrice.innerHTML += data.predicted_price.toString();
    });
}

function onPageLoad(){
    var url = "https://fp1-model-price-prediction.herokuapp.com/get_cab_names";
    $.get(url, function(data, status){
        if(data){
            var cab_names = data.cab_names;
            var uiCabName = document.getElementById('ui_cab_name');
            $('#ui_cab_name').empty();
            for (var i in cab_names){
                var opt = new Option(cab_names[i]);
                $('#ui_cab_name').append(opt);
            }
        }
    });
}
window.onload = onPageLoad;