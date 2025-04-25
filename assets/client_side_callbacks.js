
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        validateConfirmButton: function (radioSelection) {
            return radioSelection === null;
        },

        scrollToChip: function (searchValue) {
            if (searchValue === "") {
                return window.dash_clientside.no_update;
            }

            const scrollArea = document.getElementById('my-scroll-area');
            const chipInputs = scrollArea.querySelectorAll('[id^="chip-"]');

            for (const input of chipInputs) {
                const labelElement = input.nextElementSibling;
                const spanElement = labelElement.querySelector('span');
                const chipText = spanElement.textContent.trim().toLowerCase();
                if (chipText.includes(searchValue.toLowerCase())) {
                    console.log(chipText);
                    input.scrollIntoView({ block: 'center', behavior: 'smooth'});

                    const id = input.id;
                    return id.substring(id.indexOf('-') + 1);
                }
            }

            return window.dash_clientside.no_update;
        },

        triggerTrue: function(trigger) {
            if (trigger === null) {
                return window.dash_clientside.no_update;
            }

            return true;
        },

        getDpr: function(trigger) {
            if (trigger === null) {
                return window.dash_clientside.no_update;
            }

            return window.devicePixelRatio;
        }
    }
});

