/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2023 PCOpt/NTUA
    Copyright (C) 2013-2023 FOSS GP
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "adjointLaminarPowerLaw.H"
#include "IOdictionary.H"
#include "addToRunTimeSelectionTable.H"
#include "fvc.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressibleAdjoint
{
namespace adjointRASModels
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(adjointLaminarPowerLaw, 0);
addToRunTimeSelectionTable
(
    adjointRASModel,
    adjointLaminarPowerLaw,
    dictionary
);

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void adjointLaminarPowerLaw::readCoeffs()
{
    // Power-law index and clamps come from the primal transportProperties so
    // there is a single source of truth shared with libpowerLawArrhenius.
    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false                       // do not register
        )
    );
    const dictionary& plc =
        transportProperties.subDict("powerLawArrheniusCoeffs");

    n_    = dimensionedScalar("n",     dimless,      plc).value();
    nuMin_ = dimensionedScalar("nuMin", dimViscosity, plc).value();
    nuMax_ = dimensionedScalar("nuMax", dimViscosity, plc).value();

    // Linearization controls (optional sub-dict in adjointRASProperties)
    const dictionary& c = coeffDict_;
    linearizeViscosity_ = c.getOrDefault<Switch>("linearizeViscosity", true);
    linearizationSign_  = c.getOrDefault<scalar>("linearizationSign", 1);
    includeFISensitivity_ =
        c.getOrDefault<Switch>("includeFISensitivity", true);
    FISensitivityTranspose_ =
        c.getOrDefault<Switch>("FISensitivityTranspose", false);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

adjointLaminarPowerLaw::adjointLaminarPowerLaw
(
    incompressibleVars& primalVars,
    incompressibleAdjointMeanFlowVars& adjointVars,
    objectiveManager& objManager,
    const word& adjointTurbulenceModelName,
    const word& modelName
)
:
    adjointLaminar
    (
        primalVars,
        adjointVars,
        objManager,
        adjointTurbulenceModelName,
        modelName
    ),
    n_(1),
    nuMin_(0),
    nuMax_(GREAT),
    linearizeViscosity_(true),
    linearizationSign_(1),
    includeFISensitivity_(true),
    FISensitivityTranspose_(false)
{
    readCoeffs();

    Info<< "adjointLaminarPowerLaw: shear-rate viscosity linearization "
        << (linearizeViscosity_ ? "ON" : "OFF")
        << " (n = " << n_ << ", sign = " << linearizationSign_ << ")" << nl
        << "adjointLaminarPowerLaw: FI shape-sensitivity term "
        << (includeFISensitivity_ ? "ON" : "OFF")
        << (FISensitivityTranspose_ ? " (transposed)" : "") << endl;
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void adjointLaminarPowerLaw::linearizationFields
(
    tmp<volSymmTensorField>& tS,
    tmp<volScalarField>& tbeta,
    tmp<volScalarField>& tc
) const
{
    const volVectorField& U = primalVars_.U();
    const volVectorField& Ua = adjointVars_.Ua();
    tmp<volScalarField> tnu = nu();
    const volScalarField& nu = tnu();

    // Primal strain-rate tensor and magnitude gammaDot = sqrt(2 S:S)
    tS = tmp<volSymmTensorField>::New("Slin", symm(fvc::grad(U)));
    const volSymmTensorField& S = tS();

    const dimensionedScalar gammaDotSmall(dimless/dimTime, SMALL);
    const volScalarField gammaDot(max(sqrt(2.0)*mag(S), gammaDotSmall));

    // beta = (4/gammaDot) dnu/dgammaDot = 4 (n-1) nu / gammaDot^2, masked to
    // zero on the viscosity clamps (where dnu/dgammaDot = 0).
    const dimensionedScalar nuMin(dimViscosity, nuMin_);
    const dimensionedScalar nuMax(dimViscosity, nuMax_);
    const volScalarField unclamped
    (
        pos(nu - nuMin*1.0001)*pos(nuMax*0.9999 - nu)
    );

    tbeta =
        tmp<volScalarField>::New
        (
            "betaLin",
            linearizationSign_*4.0*(n_ - 1.0)*nu/sqr(gammaDot)*unclamped
        );

    // Contraction c = S : grad(Ua) (= S : symm(grad Ua) since S is symmetric)
    tc = tmp<volScalarField>::New("cLin", S && fvc::grad(Ua));
}


tmp<volVectorField> adjointLaminarPowerLaw::adjointMeanFlowSource()
{
    if (!linearizeViscosity_)
    {
        // Behave exactly as frozen-viscosity adjointLaminar
        return adjointLaminar::adjointMeanFlowSource();
    }

    tmp<volSymmTensorField> tS;
    tmp<volScalarField> tbeta;
    tmp<volScalarField> tc;
    linearizationFields(tS, tbeta, tc);

    // Extra adjoint momentum term: - div( beta c S ), added to the LHS via
    // adjointSimple's "+ adjointMeanFlowSource()".
    return volVectorField::New
    (
        "adjointMeanFlowSource" + type(),
        IOobject::NO_REGISTER,
        -fvc::div((tbeta()*tc())*tS())
    );
}


tmp<volTensorField> adjointLaminarPowerLaw::FISensitivityTerm()
{
    if (!linearizeViscosity_ || !includeFISensitivity_)
    {
        // Inherit adjointLaminar's zero -- correct for a constant viscosity,
        // and the (incomplete) behaviour of this model before the term existed.
        return adjointLaminar::FISensitivityTerm();
    }

    // Direct shape dependence of the shear-rate-dependent viscosity.
    //
    // Deforming the grid at fixed nodal field values changes the spatial
    // derivatives,
    //     d(du_k/dx_l)|_grid = - (du_k/dx_m)(d(dx_m)/dx_l)
    // hence dS|_grid = -symm(gradU & gradDx) and, through
    // dnu = (dnu/dgammaDot)(2/gammaDot) S:dS, the viscous term
    // int 2 dnu S:grad(Ua) picks up the contribution
    //
    //     - beta c (du_k/dx_m) S_kl  * d(dx_m)/dx_l
    //
    // i.e. the multiplier of grad(dx) is -beta*c*(gradU & S) in OpenFOAM's
    // gradU_ij = du_j/dx_i convention. Consumed by
    // incompressibleAdjointSolver::computeGradDxDbMultiplier as "Term 6".
    tmp<volSymmTensorField> tS;
    tmp<volScalarField> tbeta;
    tmp<volScalarField> tc;
    linearizationFields(tS, tbeta, tc);

    tmp<volTensorField> tterm
    (
        volTensorField::New
        (
            "volumeSensTerm" + type(),
            IOobject::NO_REGISTER,
            -(tbeta()*tc())*(fvc::grad(primalVars_.U()) & tS())
        )
    );

    if (FISensitivityTranspose_)
    {
        tterm.ref() = tterm().T();
    }

    return tterm;
}


bool adjointLaminarPowerLaw::read()
{
    if (adjointLaminar::read())
    {
        readCoeffs();
        return true;
    }

    return false;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace adjointRASModels
} // End namespace incompressibleAdjoint
} // End namespace Foam

// ************************************************************************* //
