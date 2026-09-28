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
#include "fvcGrad.H"

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

// * * * * * * * * * * * * * * * * Coefficients  * * * * * * * * * * * * * //

powerLawArrhenius::coeffs::coeffs(const dictionary& dict)
:
    k("k", dimViscosity, dict),
    n("n", dimless, dict),
    nuMin("nuMin", dimViscosity, dict),
    nuMax("nuMax", dimViscosity, dict),
    Eactive("Eactive", dimEnergy/dimMoles, dict),
    Rconst("Rconst", dimEnergy/(dimMoles*dimTemperature), dict)
{}


// * * * * * * * * * * * * * * Static Member Functions * * * * * * * * * * * //

tmp<volScalarField> powerLawArrhenius::calcNu
(
    const coeffs& c,
    const volVectorField& U,
    const volScalarField& T
)
{
    // same strain rate as viscosityModel::strainRate()
    const volScalarField strainRate(sqrt(2.0)*mag(symm(fvc::grad(U))));

    return max
    (
        c.nuMin,
        min
        (
            c.nuMax,
            c.k*pow
            (
                max
                (
                    dimensionedScalar("one", dimTime, 1.0)*strainRate,
                    dimensionedScalar("SMALL", dimless, SMALL)
                ),
                c.n.value() - scalar(1)
            )
            * exp(c.Eactive/(c.Rconst*T))
        )
    );
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

tmp<volScalarField> powerLawArrhenius::calcNu() const
{
    return calcNu(coeffs_, U_, U_.mesh().lookupObject<volScalarField>("T"));
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
    coeffs_(powerLawArrheniusCoeffs_),
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

    coeffs_ = coeffs(powerLawArrheniusCoeffs_);
    viscosityRelaxation_ =
        powerLawArrheniusCoeffs_.getOrDefault<scalar>("viscosityRelaxation", 1);

    return true;
}

} // namespace viscosityModels
} // namespace Foam

// ************************************************************************* //
