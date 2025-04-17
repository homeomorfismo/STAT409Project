# Data Description

Description of the dataset used in the project.

## Convention

We use `camelCase` for the names (columns) in the dataset.

- `flueDepth`: The depth of the flue (currently in `m` *meters*).
- `frequency`: The frequency of the sound (currently in `Hz` *hertz*).
- `cutUpHeight`: The height of the cut-up (currently in `m` *meters*).
- `diameterToe`: The diameter of the toe (currently in `m` *meters*).
- `acousticIntensity`: The acoustic intensity (currently in `dB` *decibels*).
- `partialN`: The `N`-th partial (currently in `dB` *decibels*).

## Known values

There are some values that are either *known*, or *computable*.
We prefix them with `ref<name>` to indicate that they are not part of the dataset, but are used in the code.
Here is a (non-exhaustive) list of those values:

- `refFrequency = 440`: The reference frequency **A4** (currently in `Hz` *hertz*).
- `refHalfStep = np.power(2, 1/12)`: The reference half step (currently in `Hz` *hertz*).
- TODO: `refAirDensity0 = 1.192`
- TODO: `refTemperature0 = 23.1`
- TODO: `refAltitude0 = 0.0` in `m` *meters*.
- TODO: `refAirDensity1 = 1.157`
- TODO: `refTemperature1 = 21.4`
- TODO: `refAltitude1 = 305.0` in `m` *meters*.

## Known formulae

- Other frequencies can be computed using the formula:
    `frequency = refFrequency * refHalfStep ** n`
    where `n` is the number of half steps from the reference frequency (can be negative).

