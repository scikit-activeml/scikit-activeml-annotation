
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        validateConfirmButton: function (radioSelection) {
            return radioSelection === null;
        },
    }
});

