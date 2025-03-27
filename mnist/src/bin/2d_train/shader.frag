#version 330

// Input uniform values
uniform vec2 weights00;
uniform vec2 weights01;
uniform vec2 weights02;
uniform vec2 weights03;
uniform vec4 biases0;

uniform vec4 weights1;
uniform float bias1;

uniform vec2 screenSize;

// Output fragment color
out vec4 finalColor;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

const float LIGHTNESS = 0.5;

void main() {
    vec2 input_vec = ((gl_FragCoord.xy/screenSize) * 8.0 - 4.0);
    input_vec.y = -input_vec.y; // fighting against raylib y axis flipping. raysan why?

    // Step 1: Compute hidden layer activation
    vec4 hidden = vec4(dot(input_vec, weights00), dot(input_vec, weights01), dot(input_vec, weights02), dot(input_vec, weights03)) + biases0;
    hidden = vec4(sigmoid(hidden.x), sigmoid(hidden.y), sigmoid(hidden.z), sigmoid(hidden.w));

    // Step 2: Compute output neuron activation
    float net_output = dot(weights1, hidden) + bias1;
    net_output = sigmoid(net_output);

    // Use output to color pixels
    if (net_output > 0.5) {
        finalColor = vec4(LIGHTNESS, LIGHTNESS, 1.0, 1.0);
    } else {
        finalColor = vec4(1.0, LIGHTNESS, LIGHTNESS, 1.0);
    }
}