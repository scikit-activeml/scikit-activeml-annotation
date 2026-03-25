window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        validateConfirmButton: function (radioSelection) {
            return radioSelection === null;
        },

        scrollToChip: function (searchValue) {
            function selectChip(node, chipText) {
                node.scrollIntoView({ block: 'center', behavior: 'smooth' });
                console.log("scroll to Chip with text: " + chipText)
                return chipText;
            }

            if (searchValue === "") {
                return window.dash_clientside.no_update;
            }

            const search = searchValue.trim().toLowerCase();
            const scrollArea = document.getElementById('my-scroll-area');

            if (!scrollArea) {
                console.error("scrollToChip: could not find scrollArea with id 'my-scroll-area'");
                return window.dash_clientside.no_update;
            }

            // Find all children of scrollarea which ids starts with chip-
            const chipInputs = scrollArea.querySelectorAll('[id^="chip-"]');

            let prefixMatchNode = null;
            let prefixMatchText = null;

            for (const input of chipInputs) {
                const id = input.id;
                // Infer chipText from id. Expects id to be of format chip-{label}
                const chipText = id.substring(id.indexOf('-') + 1);
                const chipTextLower = chipText.toLowerCase();

                // Exact match
                if (chipTextLower === search) {
                    return selectChip(input, chipText);
                }

                // Prefix match
                if (!prefixMatchNode && chipTextLower.startsWith(search)) {
                    prefixMatchNode = input;
                    prefixMatchText = chipText;
                }
            }

            // There was no exact match but a prefix match
            if (prefixMatchNode) {
                return selectChip(prefixMatchNode, prefixMatchText);
            }

            // No Match found
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
        },

        disableAllButtons: function(n_clicks_list) {
            // n_clicks_list is an array of click counts for each matched button
            if (!n_clicks_list) {
                return window.dash_clientside.no_update;
            }

            // If any button has been clicked (any value > 0 or truthy)
            if (n_clicks_list.some(n => n)) {
                // Disable all buttons
                return Array(n_clicks_list.length).fill(true);
            }

            // Otherwise, keep them enabled
            return Array(n_clicks_list.length).fill(false);
        },

        goToLastPage: function(n_clicks) {
            // Use browser api to go back to last page
            window.history.back();
        },

        clickButtonWithId: function(btn_id) {
            const el = window.dash_clientside.helpers.getElementFromDashId(btn_id);
            if (!el) {
                console.warn("Cannot click button as no element found with ID:", btn_id);
                return;
            }
            el.click();
        },

        focusElementWithId: function(el_id) {
            const el = window.dash_clientside.helpers.getElementFromDashId(el_id);
            if (!el) {
                console.warn("Cannot focus element as no element is found with ID:", el_id);
                return;
            }
            el.focus();
        },

    },

    helpers: {
        getElementFromDashId: function(dashId) {
            if (!dashId) return null;

            let domId;

            if (typeof dashId === "object") {
                // If using dash Pattern matching the id is an object
                // Convert it to a string as this string must be present in the dom.

                // Dash sorts keys alphabetically when serializing dict IDs into DOM
                // need to replicate that here so the ids match
                const keys = Object.keys(dashId).sort();
                domId = JSON.stringify(dashId, keys);
            } else if (typeof dashId === "string") {
                domId = dashId;
            } else {
                console.error("Invalid ID type it must be string or Object but is:", typeof dashId, dashId);
                return null;
            }

            const el = document.getElementById(domId);
            if (!el) {
                console.error("No element found with ID:", domId);
                return null;
            }

            return el;
        },
    },
});

