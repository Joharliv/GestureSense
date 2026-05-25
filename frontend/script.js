const video = document.getElementById("video");

const predictionText =
    document.getElementById("prediction");

const predictionType =
    document.getElementById("predictionType");

const sentenceText =
    document.getElementById("sentence");

const suggestionsText =
    document.getElementById("suggestions");

const startBtn =
    document.getElementById("startBtn");

const stopBtn =
    document.getElementById("stopBtn");

const modeButtons =
    document.querySelectorAll(".mode-btn");

const builderUI =
    document.getElementById("builderUI");

const deleteLetterBtn =
    document.getElementById("deleteLetterBtn");

const deleteWordBtn =
    document.getElementById("deleteWordBtn");

const clearSentenceBtn =
    document.getElementById("clearSentenceBtn");

const API_BASE =
    "https://gesturesense-zf4a.onrender.com";


// ======================================
// CREATE SPACE BUTTON
// ======================================

const spaceBtn =
    document.createElement("button");

spaceBtn.id = "spaceBtn";

spaceBtn.className =
    "sentence-btn";

spaceBtn.innerText =
    "␣ Space";

const actionsContainer =
    document.querySelector(
        ".sentence-actions"
    );

actionsContainer.insertBefore(
    spaceBtn,
    deleteLetterBtn
);


// ======================================
// VARIABLES
// ======================================

let stream = null;

let interval = null;

let currentMode = "motion";

let currentSentence = "";

let currentPrediction = "";

let stableFrames = 0;

let isProcessing = false;

let canvas =
    document.createElement("canvas");

let ctx =
    canvas.getContext("2d");


// ======================================
// INITIAL UI
// ======================================

builderUI.style.display = "none";


// ======================================
// MODE SWITCH
// ======================================

modeButtons.forEach(button => {

    button.addEventListener("click", () => {

        modeButtons.forEach(btn => {

            btn.classList.remove(
                "active-mode"
            );
        });

        button.classList.add(
            "active-mode"
        );

        currentMode =
            button.dataset.mode;

        if(currentMode === "builder"){

            builderUI.style.display =
                "flex";

            builderUI.style.flexDirection =
                "column";

            builderUI.style.gap =
                "18px";
        }

        else{

            builderUI.style.display =
                "none";
        }

        predictionText.innerText =
            "Waiting...";

        predictionType.innerText =
            currentMode.toUpperCase();

        suggestionsText.innerHTML = "";

        currentPrediction = "";

        stableFrames = 0;
    });
});


// ======================================
// START CAMERA
// ======================================

async function startCamera(){

    try{

        stream =
            await navigator
            .mediaDevices
            .getUserMedia({

                video:true,
                audio:false
            });

        video.srcObject = stream;

        await video.play();

        canvas.width =
            video.videoWidth;

        canvas.height =
            video.videoHeight;

        interval =
            setInterval(
                sendFrame,
                150
            );
    }

    catch(error){

        console.log(error);
    }
}


// ======================================
// STOP CAMERA
// ======================================

function stopCamera(){

    if(stream){

        stream.getTracks().forEach(track => {

            track.stop();
        });

        stream = null;
    }

    clearInterval(interval);

    interval = null;

    video.srcObject = null;

    predictionText.innerText =
        "Camera Stopped";
}


// ======================================
// SEND FRAME
// ======================================

async function sendFrame(){

    if(isProcessing) return;

    if(video.readyState !== 4) return;

    isProcessing = true;

    try{

        ctx.drawImage(

            video,

            0,
            0,

            canvas.width,
            canvas.height
        );

        const image =
            canvas.toDataURL(
                "image/jpeg",
                0.6
            );

        const response = await fetch(
          `${API_BASE}/predict/${currentMode}`,

          {
            method: "POST",

            headers: {
              "Content-Type": "application/json",
            },

            body: JSON.stringify({
              image: image,
            }),
          },
        );

        const data =
            await response.json();

        const pred = (data.prediction ?? data ?? "").toString().trim();

        predictionText.innerText =
            pred || "...";

        predictionType.innerText =
            currentMode.toUpperCase();


        // ======================================
        // NORMAL MODES
        // ======================================

        if(

            currentMode === "letter" ||

            currentMode === "word"

        ){

            return;
        }


        // ======================================
        // BUILDER MODE
        // ======================================

        const invalid = [

            "",
            "...",
            "none",
            "no hand"
        ];

        if(

            invalid.includes(
                pred.toLowerCase()
            )

        ){

            currentPrediction = "";

            stableFrames = 0;

            return;
        }


        // ======================================
        // STABILITY CHECK
        // ======================================

        if(pred === currentPrediction){

            stableFrames++;
        }

        else{

            currentPrediction = pred;

            stableFrames = 1;
        }


        // ======================================
        // ACCEPT LETTER
        // ======================================

        if(stableFrames === 5){

    // WORD MODE INSIDE BUILDER

    if(data.type === "WORD"){

        currentSentence += pred + " ";
    }

    // LETTER MODE INSIDE BUILDER

    else{

        currentSentence += pred;
    }

    sentenceText.innerText =
        currentSentence;
}


        // ======================================
        // SUGGESTIONS
        // ======================================

        suggestionsText.innerHTML = "";

        if(data.suggestions){

            data.suggestions.forEach(suggestion => {

                const tag =
                    document.createElement("span");

                tag.className =
                    "suggestion-tag";

                tag.innerText =
                    suggestion;

                // ==========================
                // ADD SUGGESTION
                // ==========================

                tag.onclick = () => {

                    let words =
                        currentSentence
                        .trim()
                        .split(" ");

                    words.pop();

                    currentSentence =
                        words.join(" ");

                    if(

                        currentSentence.length > 0 &&

                        !currentSentence.endsWith(" ")

                    ){

                        currentSentence += " ";
                    }

                    currentSentence +=
                        suggestion + " ";

                    sentenceText.innerText =
                        currentSentence;

                    suggestionsText.innerHTML =
                        "";
                };

                suggestionsText.appendChild(tag);
            });
        }
    }

    catch(error){

        console.log(error);
    }

    finally{

        isProcessing = false;
    }
}


// ======================================
// DELETE LETTER
// ======================================

deleteLetterBtn.addEventListener("click", () => {

    currentSentence =
        currentSentence.slice(0, -1);

    sentenceText.innerText =
        currentSentence;
});


// ======================================
// DELETE WORD
// ======================================

deleteWordBtn.addEventListener("click", () => {

    let words =
        currentSentence
        .trim()
        .split(" ");

    words.pop();

    currentSentence =
        words.join(" ");

    sentenceText.innerText =
        currentSentence;
});


// ======================================
// CLEAR
// ======================================

clearSentenceBtn.addEventListener("click", () => {

    currentSentence = "";

    sentenceText.innerText = "";

    suggestionsText.innerHTML = "";
});


// ======================================
// SPACE BUTTON
// ======================================

spaceBtn.addEventListener("click", () => {

    if(

        currentSentence.length > 0 &&

        !currentSentence.endsWith(" ")

    ){

        currentSentence += " ";

        sentenceText.innerText =
            currentSentence;
    }
});


// ======================================
// BUTTON EVENTS
// ======================================

startBtn.addEventListener(
    "click",
    startCamera
);

stopBtn.addEventListener(
    "click",
    stopCamera
);