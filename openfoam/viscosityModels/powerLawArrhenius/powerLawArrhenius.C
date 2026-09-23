/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2017 OpenCFD Ltd
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.
\*---------------------------------------------------------------------------*/

#include "powerLawArrhenius.H"
#include "addToRunTimeSelectionTable.H"
#include "surfaceFields.H"

namespace Foam
{
namespace viscosityModels
{
    defineTypeNameAndDebug(powerLawArrhenius, 0);

    addToRunTimeSelectionTable
    (
        viscosityModel,
        powerLawArrhenius,
        dictionary
    );

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField>

Foam::viscosityModels::powerLawArrhenius::calcNu() const
{
    const volScalarField& T=U_.mesh().lookupObject<volScalarField>("T");
    return max
    (
        nuMin_,
        min
        (
            nuMax_,
            k_*pow 
            (
                max
                (
                    dimensionedScalar("one", dimTime, 1.0)*strainRate(),
                    dimensionedScalar("SMALL", dimless, SMALL)
                ),
                n_.value() - scalar(1)
            )
            * exp( Eactive_ / (Rconst_ * T) )
        )
    );


}




// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

powerLawArrhenius::powerLawArrhenius
(
    const word& name,
    const dictionary& viscosityProperties,
    const volVectorField& U,
    const surfaceScalarField& phi
)
:
    viscosityModel(name, viscosityProperties, U, phi),
    powerLawArrheniusCoeffs_(viscosityProperties.optionalSubDict(typeName + "Coeffs")),
    k_("k", dimViscosity, powerLawArrheniusCoeffs_),
    n_("n", dimless, powerLawArrheniusCoeffs_),
    nuMin_("nuMin", dimViscosity, powerLawArrheniusCoeffs_),
    nuMax_("nuMax", dimViscosity, powerLawArrheniusCoeffs_),
    Eactive_("Eactive", dimEnergy/dimMoles, powerLawArrheniusCoeffs_),                             // [J/mol]
    Rconst_("Rconst",  dimEnergy/(dimMoles*dimTemperature), powerLawArrheniusCoeffs_),            // [J/(mol K)]
    viscosityRelaxation_
    (
        powerLawArrheniusCoeffs_.getOrDefault<scalar>("viscosityRelaxation", 1)
    ),
    nu_
    (
        IOobject
        (
            name,
            U_.time().timeName(),
            U_.db(),
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        calcNu()
    )
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool powerLawArrhenius::read
(
    const dictionary& viscosityProperties
)
{
    viscosityModel::read(viscosityProperties);

    powerLawArrheniusCoeffs_ = viscosityProperties.optionalSubDict(typeName + "Coeffs");

    powerLawArrheniusCoeffs_.readEntry("k", k_);
    powerLawArrheniusCoeffs_.readEntry("n", n_);
    powerLawArrheniusCoeffs_.readEntry("nuMin", nuMin_);
    powerLawArrheniusCoeffs_.readEntry("nuMax", nuMax_);
    powerLawArrheniusCoeffs_.readEntry("Eactive", Eactive_);
    powerLawArrheniusCoeffs_.readEntry("Rconst",  Rconst_);
    viscosityRelaxation_ =
        powerLawArrheniusCoeffs_.getOrDefault<scalar>("viscosityRelaxation", 1);

    return true;
}

} // namespace viscosityModels
} // namespace Foam

// ************************************************************************* //
